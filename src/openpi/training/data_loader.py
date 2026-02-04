from collections.abc import Iterator, Sequence
import multiprocessing
import os
import psutil
import time
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms
from openpi.training.data_loading_monitor import (
    DataLoadingMonitor,
    DataLoadingMetrics,
    get_global_monitor,
)

T_co = TypeVar("T_co", covariant=True)


def _get_optimal_num_workers(dataset_size: int, batch_size: int) -> int:
    """Determine optimal number of workers based on system resources.
    
    Args:
        dataset_size: Size of the dataset
        batch_size: Batch size being used
        
    Returns:
        Optimal number of worker processes
    """
    # Get system information
    cpu_count = multiprocessing.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Base calculation: use 2-4 workers per CPU core, but cap based on memory
    # Each worker typically uses 100-500MB of memory depending on dataset
    max_workers_by_cpu = min(cpu_count * 2, 16)  # Cap at 16 workers
    max_workers_by_memory = max(1, int(available_memory_gb / 0.5))  # Assume 500MB per worker
    
    # Consider dataset characteristics
    # For small datasets, fewer workers are better to avoid overhead
    if dataset_size < 1000:
        max_workers_by_dataset = 2
    elif dataset_size < 10000:
        max_workers_by_dataset = 4
    else:
        max_workers_by_dataset = 8
    
    # Take the minimum of all constraints
    optimal_workers = min(max_workers_by_cpu, max_workers_by_memory, max_workers_by_dataset)
    
    # Always use at least 1 worker for better GPU utilization, but allow 0 for debugging
    return max(1, optimal_workers)


def _get_optimal_prefetch_factor(num_workers: int) -> int:
    """Determine optimal prefetch factor based on number of workers.
    
    Args:
        num_workers: Number of worker processes
        
    Returns:
        Optimal prefetch factor per worker
    """
    if num_workers == 0:
        return 2  # Default PyTorch value for single-threaded
    elif num_workers <= 2:
        return 4  # Higher prefetch for few workers
    elif num_workers <= 4:
        return 3  # Moderate prefetch for medium workers
    else:
        return 2  # Lower prefetch for many workers to avoid memory pressure


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Optional: use gr00t local LeRobot dataset loader instead of HF LeRobotDataset.
    if data_config.gr00t_dataset_path is not None:
        from openpi.training.gr00t_lerobot_dataset import Gr00tDatasetSpec, Gr00tLeRobotTorchDataset

        if data_config.gr00t_embodiment_tag is None:
            raise ValueError("gr00t_embodiment_tag must be set when gr00t_dataset_path is set.")

        # Note: action_horizon is implied by gr00t modality config delta_indices. We do a soft check here
        # to catch accidental mismatches between model config and embodiment config.
        # The dataset will still return whatever horizon the embodiment config defines.
        
        # Apply memory optimizations for large datasets or limited GPU memory
        enable_memory_opt = getattr(data_config, 'gr00t_enable_memory_optimization', False)
        max_cache_size = getattr(data_config, 'gr00t_max_video_cache_size_mb', 256)
        
        spec = Gr00tDatasetSpec(
            dataset_path=data_config.gr00t_dataset_path,
            embodiment_tag=data_config.gr00t_embodiment_tag,
            modality_config_path=data_config.gr00t_modality_config_path,
            action_horizon=int(action_horizon),
            language_key=data_config.gr00t_language_key,
            video_views=list(data_config.gr00t_video_views) if data_config.gr00t_video_views is not None else None,
            episode_cache_size=int(data_config.gr00t_episode_cache_size),
            video_backend=data_config.gr00t_video_backend,
            apply_action_transforms=data_config.gr00t_apply_action_transforms,
            stats_key=data_config.gr00t_stats_key,
            # Memory optimization settings
            enable_memory_optimization=enable_memory_opt,
            max_video_cache_size_mb=max_cache_size,
            video_frame_compression=getattr(data_config, 'gr00t_video_frame_compression', False),
            video_frame_quality=getattr(data_config, 'gr00t_video_frame_quality', 85),
            lazy_video_loading=getattr(data_config, 'gr00t_lazy_video_loading', False),
            reduce_video_resolution=getattr(data_config, 'gr00t_reduce_video_resolution', False),
            target_video_resolution=getattr(data_config, 'gr00t_target_video_resolution', (224, 224)),
            enable_frame_skipping=getattr(data_config, 'gr00t_enable_frame_skipping', False),
            frame_skip_factor=getattr(data_config, 'gr00t_frame_skip_factor', 1),
        )
        dataset = Gr00tLeRobotTorchDataset(spec)
        return dataset

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    auto_optimize_workers: bool = True,
    enable_monitoring: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    # When GR00T action transforms are enabled, skip OpenPI's normalization
    # since StateActionProcessor handles all normalization
    if hasattr(data_config, 'gr00t_apply_action_transforms') and data_config.gr00t_apply_action_transforms:
        skip_norm_stats = True

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        auto_optimize_workers=auto_optimize_workers,
        enable_monitoring=enable_monitoring,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    auto_optimize_workers: bool = True,
    enable_monitoring: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process. If auto_optimize_workers is True, this acts as
            a maximum constraint.
        seed: The seed to use for shuffling the data.
        auto_optimize_workers: Whether to automatically optimize worker count and prefetch
            settings based on system resources.
        enable_monitoring: Whether to enable performance monitoring and bottleneck detection.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        auto_optimize=auto_optimize_workers,
        enable_monitoring=enable_monitoring,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        auto_optimize: bool = True,
        enable_monitoring: bool = True,
    ):
        """Create a PyTorch data loader with GPU utilization optimizations.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process. If auto_optimize is True, this will be used as
                a maximum constraint.
            seed: The seed to use for shuffling the data.
            auto_optimize: If True, automatically optimize worker count and prefetch settings
                based on system resources and dataset characteristics.
            enable_monitoring: If True, enable performance monitoring and bottleneck detection.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches
        self._local_batch_size = local_batch_size
        self._enable_monitoring = enable_monitoring

        # Initialize performance monitor
        self._monitor = get_global_monitor() if enable_monitoring else None

        # Optimize worker count and prefetch settings if requested
        if auto_optimize:
            optimal_workers = _get_optimal_num_workers(len(dataset), local_batch_size)
            # Use the minimum of requested workers and optimal workers
            if num_workers > 0:
                num_workers = min(num_workers, optimal_workers)
            else:
                num_workers = optimal_workers
        
        # Store optimization settings for monitoring
        self._num_workers = num_workers
        self._auto_optimize = auto_optimize

        mp_context = None
        prefetch_factor = 2  # Default PyTorch value
        pin_memory = False
        persistent_workers = False
        
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")
            # Use optimized prefetch factor for better GPU utilization
            prefetch_factor = _get_optimal_prefetch_factor(num_workers)
            # Disable memory pinning in containerized environments to avoid hangs
            pin_memory = False  # Changed from True to fix container hangs
            # Disable persistent workers to avoid process management issues in containers
            persistent_workers = False  # Changed from True to fix container hangs

        generator = torch.Generator()
        generator.manual_seed(seed)
        
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader
    
    def get_optimization_info(self) -> dict[str, any]:
        """Get information about the optimization settings used."""
        return {
            "num_workers": self._num_workers,
            "prefetch_factor": self._data_loader.prefetch_factor,
            "pin_memory": self._data_loader.pin_memory,
            "persistent_workers": self._data_loader.persistent_workers,
            "auto_optimize": self._auto_optimize,
            "enable_monitoring": self._enable_monitoring,
        }

    def get_performance_stats(self) -> dict[str, any]:
        """Get current performance statistics."""
        if self._monitor is not None:
            return self._monitor.get_current_stats()
        return {}

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    # Record batch loading time
                    batch_start_time = time.time()
                    batch = next(data_iter)
                    batch_load_time = time.time() - batch_start_time
                    
                    # Record performance metrics
                    if self._monitor is not None:
                        # Get video decode stats if available (from GR00T dataset)
                        video_decode_time = 0.0
                        cache_hit_rate = -1.0  # -1 indicates no cache info
                        
                        # Try to get video stats from dataset
                        if hasattr(self._data_loader.dataset, 'get_video_decode_stats'):
                            video_stats = self._data_loader.dataset.get_video_decode_stats()
                            if video_stats:
                                cache_hit_rate = video_stats.get('cache_hit_rate', -1.0)
                                # Estimate video decode time from recent operations
                                if 'decode_time_avg' in video_stats:
                                    video_decode_time = video_stats['decode_time_avg']
                        
                        metrics = DataLoadingMetrics(
                            batch_load_time=batch_load_time,
                            batch_size=self._local_batch_size,
                            video_decode_time=video_decode_time,
                            cache_hit_rate=cache_hit_rate,
                        )
                        self._monitor.record_batch_metrics(metrics)
                    
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
