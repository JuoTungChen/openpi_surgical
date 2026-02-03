import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.jax_optimizer as _jax_optimizer
import openpi.training.multi_gpu_coordinator as _multi_gpu
import openpi.training.optimization_config as _opt_config
import openpi.training.optimizer as _optimizer
import openpi.training.performance_integration as _performance
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def create_optimization_configs(config: _config.TrainConfig) -> tuple[
    _performance.PerformanceIntegrationConfig,
    _jax_optimizer.JaxOptimizationConfig,
    _multi_gpu.MultiGPUConfig,
]:
    """Create optimization configurations with graceful fallback for disabled features."""
    
    # Create comprehensive optimization config from train config
    try:
        # Check if user wants to disable all optimizations
        if getattr(config, 'disable_all_optimizations', False):
            opt_config = _opt_config.OptimizationConfig()
            opt_config.disable_all_optimizations()
            logging.info("All optimizations disabled by user request")
        else:
            # Create config based on hardware setup and optimization level
            hardware_setup = getattr(config, 'hardware_setup', 'auto')
            opt_config = _opt_config.OptimizationConfig.create_for_hardware_setup(hardware_setup)
            
            # Override optimization level if specified
            optimization_level = getattr(config, 'optimization_level', 'balanced')
            opt_config.optimization_level = optimization_level
            
            # Apply optimization level settings
            if optimization_level == "conservative":
                opt_config._apply_conservative_settings()
            elif optimization_level == "aggressive":
                opt_config._apply_aggressive_settings()
            
            # Override with any explicit train config settings
            opt_config = _opt_config.create_optimization_config_from_train_config(config)
            
            logging.info(f"Created optimization config with level: {opt_config.optimization_level}, setup: {hardware_setup}")
        
        # Log effective configuration
        effective_config = opt_config.get_effective_config_dict()
        logging.info("Effective optimization configuration:")
        for key, value in effective_config.items():
            if isinstance(value, dict):
                logging.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (bool, int, float, str)):
                        logging.info(f"    {sub_key}: {sub_value}")
            else:
                logging.info(f"  {key}: {value}")
    
    except Exception as e:
        logging.warning(f"Failed to create optimization config, using defaults: {e}")
        opt_config = _opt_config.OptimizationConfig()
        opt_config.disable_all_optimizations()
    
    # Performance monitoring configuration with graceful fallback
    try:
        performance_config = _performance.PerformanceIntegrationConfig(
            target_gpu_utilization=opt_config.gpu_monitoring.target_gpu_utilization,
            monitoring_interval=opt_config.gpu_monitoring.monitoring_interval,
            enable_gpu_monitoring=opt_config.gpu_monitoring.enable_gpu_monitoring,
            enable_bottleneck_detection=opt_config.gpu_monitoring.enable_bottleneck_detection,
            enable_automatic_suggestions=opt_config.gpu_monitoring.enable_performance_suggestions,
            enable_wandb_logging=opt_config.gpu_monitoring.enable_wandb_logging,
            performance_log_interval=opt_config.gpu_monitoring.performance_log_interval,
            wandb_log_interval=opt_config.gpu_monitoring.wandb_log_interval,
        )
    except Exception as e:
        logging.warning(f"Failed to create performance config, using defaults: {e}")
        performance_config = _performance.PerformanceIntegrationConfig(
            enable_gpu_monitoring=False,
            enable_bottleneck_detection=False,
            enable_automatic_suggestions=False,
            enable_wandb_logging=config.wandb_enabled,
        )
    
    # JAX optimization configuration with graceful fallback
    try:
        jax_optimization_config = _jax_optimizer.JaxOptimizationConfig(
            compilation=_jax_optimizer.CompilationCacheConfig(
                enable_cache_warming=opt_config.jax_optimization.enable_jit_cache_warming,
                warmup_iterations=opt_config.jax_optimization.jit_warmup_iterations,
                persist_cache=opt_config.jax_optimization.jit_cache_persistence,
                max_cache_size=opt_config.jax_optimization.jit_max_cache_size,
                log_compilation_timing=opt_config.jax_optimization.log_compilation_timing,
            ),
            memory=_jax_optimizer.MemoryOptimizationConfig(
                enable_memory_optimization=opt_config.jax_optimization.enable_memory_optimization,
                target_memory_utilization=opt_config.jax_optimization.target_memory_utilization,
                enable_auto_batch_sizing=opt_config.jax_optimization.enable_auto_batch_sizing,
                min_batch_size=opt_config.jax_optimization.min_batch_size,
                max_batch_size=opt_config.jax_optimization.max_batch_size,
            ),
            training=_jax_optimizer.TrainingOptimizationConfig(
                enable_gradient_accumulation=opt_config.jax_optimization.enable_gradient_accumulation,
                gradient_accumulation_steps=opt_config.jax_optimization.gradient_accumulation_steps,
                enable_overlapped_computation=opt_config.jax_optimization.enable_overlapped_computation,
                enable_mixed_precision=opt_config.jax_optimization.enable_mixed_precision,
                optimize_gradient_sync=opt_config.jax_optimization.optimize_gradient_sync,
            ),
        )
    except Exception as e:
        logging.warning(f"Failed to create JAX optimization config, using defaults: {e}")
        jax_optimization_config = _jax_optimizer.JaxOptimizationConfig(
            compilation=_jax_optimizer.CompilationCacheConfig(enable_cache_warming=False),
            memory=_jax_optimizer.MemoryOptimizationConfig(enable_memory_optimization=False),
            training=_jax_optimizer.TrainingOptimizationConfig(
                enable_gradient_accumulation=False,
                enable_overlapped_computation=False,
                enable_mixed_precision=False,
                optimize_gradient_sync=False,
            ),
        )
    
    # Multi-GPU configuration with graceful fallback
    try:
        multi_gpu_config = _multi_gpu.MultiGPUConfig(
            fsdp_devices=opt_config.multi_gpu.fsdp_devices,
            enable_dynamic_device_count=opt_config.multi_gpu.enable_dynamic_device_count,
            log_sharding_decisions=opt_config.multi_gpu.log_sharding_decisions,
            enable_load_balancing=opt_config.multi_gpu.enable_load_balancing,
            communication_backend=opt_config.multi_gpu.communication_backend,
            min_shard_size_mb=opt_config.multi_gpu.min_shard_size_mb,
            memory_threshold=opt_config.multi_gpu.memory_threshold,
        )
    except Exception as e:
        logging.warning(f"Failed to create multi-GPU config, using defaults: {e}")
        multi_gpu_config = _multi_gpu.MultiGPUConfig(
            fsdp_devices=config.fsdp_devices,
            enable_dynamic_device_count=False,
            log_sharding_decisions=False,
            enable_load_balancing=False,
        )
    
    return performance_config, jax_optimization_config, multi_gpu_config


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # Create optimization configurations with graceful fallback
    performance_config, jax_optimization_config, multi_gpu_config = create_optimization_configs(config)
    
    # Initialize performance monitoring with graceful fallback
    performance_integrator = None
    try:
        performance_integrator = _performance.PerformanceIntegrator(performance_config)
        performance_integrator.start_monitoring()
        logging.info("Performance monitoring initialized successfully")
    except Exception as e:
        logging.warning(f"Failed to initialize performance monitoring: {e}")
        logging.info("Continuing without performance monitoring")

    # Initialize JAX optimization with graceful fallback
    jax_optimizer = None
    try:
        jax_optimizer = _jax_optimizer.JaxTrainingOptimizer(jax_optimization_config)
        logging.info("JAX optimization initialized successfully")
    except Exception as e:
        logging.warning(f"Failed to initialize JAX optimization: {e}")
        logging.info("Continuing without JAX optimizations")

    # Initialize multi-GPU coordinator with graceful fallback
    multi_gpu_coordinator = None
    try:
        if config.fsdp_devices > 1:
            multi_gpu_coordinator = _multi_gpu.MultiGPUCoordinator(multi_gpu_config)
            logging.info(f"Multi-GPU coordinator initialized for {config.fsdp_devices} devices")
    except Exception as e:
        logging.warning(f"Failed to initialize multi-GPU coordinator: {e}")
        logging.info("Continuing with standard FSDP sharding")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Mark data loading start for performance monitoring
    if performance_integrator:
        performance_integrator.on_data_loading_start()
    
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    
    # Mark data loading end
    if performance_integrator:
        performance_integrator.on_data_loading_end()
    
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    if config.wandb_enabled:
        try:
            images_to_log = [
                wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
                for i in range(min(5, len(next(iter(batch[0].images.values())))))
            ]
            wandb.log({"camera_views": images_to_log}, step=0)
        except Exception as e:
            logging.warning(f"Failed to log images to wandb: {e}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # Create optimized training step with compilation cache warming (with fallback)
    if jax_optimizer:
        try:
            ptrain_step = jax_optimizer.create_optimized_train_step(
                train_step_fn=train_step,
                sample_batch=batch,
                train_state=train_state,
                rng=train_rng,
                train_config=config,
                mesh=mesh,
                train_state_sharding=train_state_sharding,
                data_sharding=data_sharding,
                replicated_sharding=replicated_sharding,
            )
            logging.info("Using optimized training step")
        except Exception as e:
            logging.warning(f"Failed to create optimized training step: {e}")
            logging.info("Falling back to standard training step")
            ptrain_step = jax.jit(
                train_step,
                static_argnums=(0,),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )
    else:
        # Fallback to standard JIT compilation
        ptrain_step = jax.jit(
            train_step,
            static_argnums=(0,),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        logging.info("Using standard training step (no optimizations)")

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    try:
        for step in pbar:
            # Mark training step start for performance monitoring
            if performance_integrator:
                performance_integrator.on_training_step_start(step)
            
            # Mark data loading start
            if performance_integrator:
                performance_integrator.on_data_loading_start()
            batch = next(data_iter)
            
            # Optimize data transfer for better memory efficiency (with fallback)
            if jax_optimizer:
                try:
                    batch = jax_optimizer.optimize_data_transfer(batch, data_sharding)
                except Exception as e:
                    logging.warning(f"Data transfer optimization failed: {e}")
            
            if performance_integrator:
                performance_integrator.on_data_loading_end()
            
            # Mark computation start
            if performance_integrator:
                performance_integrator.on_computation_start()
            
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(config, train_rng, train_state, batch)
            
            # Mark computation end
            if performance_integrator:
                performance_integrator.on_computation_end()
            
            infos.append(info)
            
            # Handle logging
            if step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                if config.wandb_enabled:
                    wandb.log(reduced_info, step=step)
                infos = []
            
            # Performance monitoring step end
            if performance_integrator:
                step_info = {"loss": float(info.get("loss", 0)), "grad_norm": float(info.get("grad_norm", 0))}
                performance_integrator.on_training_step_end(step, step_info)

            # Memory monitoring and optimization (with fallback)
            if jax_optimizer and step % (config.log_interval * 5) == 0:  # Check memory every 5 log intervals
                try:
                    is_memory_safe, memory_info = jax_optimizer.memory_optimizer.check_memory_usage(config.batch_size)
                    
                    if not is_memory_safe:
                        logging.warning(f"High memory usage detected: {memory_info.get('max_utilization', 0):.2%}")
                        
                        if getattr(config, 'enable_auto_batch_sizing', False):
                            suggested_batch_size = jax_optimizer.memory_optimizer.suggest_batch_size(
                                config.batch_size, memory_info
                            )
                            if suggested_batch_size and suggested_batch_size != config.batch_size:
                                logging.info(f"Suggested batch size adjustment: {config.batch_size} -> {suggested_batch_size}")
                    
                    # Log memory stats to wandb
                    if config.wandb_enabled and step % config.log_interval == 0:
                        memory_log = {
                            f"memory/{k}": v for k, v in memory_info.items() 
                            if isinstance(v, (int, float))
                        }
                        wandb.log(memory_log, step=step)
                except Exception as e:
                    logging.warning(f"Memory monitoring failed: {e}")

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    finally:
        # Stop performance monitoring
        if performance_integrator:
            try:
                performance_integrator.stop_monitoring()
                
                # Log final performance summary
                final_summary = performance_integrator.get_performance_summary()
                logging.info("=== Final Performance Summary ===")
                for key, value in final_summary.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"{key}: {value}")
                logging.info("=" * 40)
            except Exception as e:
                logging.warning(f"Failed to get performance summary: {e}")
        
        # Log JAX optimization statistics
        if jax_optimizer:
            try:
                jax_stats = jax_optimizer.get_optimization_stats()
                logging.info("=== JAX Optimization Summary ===")
                compilation_stats = jax_stats.get("compilation", {})
                if compilation_stats:
                    logging.info(f"Cached functions: {compilation_stats.get('cached_functions', [])}")
                    logging.info(f"Total compilation time: {compilation_stats.get('total_compilation_time', 0):.2f}s")
                    logging.info(f"Cache size: {compilation_stats.get('cache_size', 0)}")
                
                training_stats = jax_stats.get("training", {})
                if training_stats:
                    logging.info(f"Gradient accumulation: {training_stats.get('gradient_accumulation_enabled', False)}")
                    if training_stats.get('gradient_accumulation_enabled'):
                        logging.info(f"Accumulation steps: {training_stats.get('gradient_accumulation_steps', 1)}")
                    logging.info(f"Overlapped computation: {training_stats.get('overlapped_computation_enabled', False)}")
                    logging.info(f"Mixed precision: {training_stats.get('mixed_precision_enabled', False)}")
                logging.info("=" * 40)
            except Exception as e:
                logging.warning(f"Failed to get JAX optimization stats: {e}")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
