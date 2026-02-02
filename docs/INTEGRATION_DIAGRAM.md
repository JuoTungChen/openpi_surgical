# GR00T-OpenPI Integration: Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  USER                                        │
│                                                                              │
│  Wants to train OpenPI using GR00T's exact data loading setup              │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                │ Provides
                                │
                    ┌───────────▼──────────┐
                    │  GR00T Config Files  │
                    │                      │
                    │  1. YAML Config      │
                    │  2. Modality Config  │
                    │  3. Statistics       │
                    └───────────┬──────────┘
                                │
                                │ Load with
                                │
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER (NEW)                                   │
│                                                                              │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃ gr00t_config_loader.py                                              ┃  │
│  ┃                                                                      ┃  │
│  ┃  load_gr00t_config(yaml, modality_config)                          ┃  │
│  ┃         │                                                           ┃  │
│  ┃         ├─► Load YAML config                                       ┃  │
│  ┃         ├─► Import modality config Python module                   ┃  │
│  ┃         ├─► Extract embodiment settings                            ┃  │
│  ┃         └─► Load normalization statistics                          ┃  │
│  ┃                 │                                                   ┃  │
│  ┃                 ▼                                                   ┃  │
│  ┃         Gr00tConfigBundle                                          ┃  │
│  ┃         ├─► get_action_horizon(embodiment) → int                   ┃  │
│  ┃         ├─► get_action_dim(embodiment) → int                       ┃  │
│  ┃         ├─► get_video_views(embodiment) → list[str]               ┃  │
│  ┃         └─► to_openpi_config(...) → TrainConfig                    ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                                              │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃ train_with_groot_config.py                                          ┃  │
│  ┃                                                                      ┃  │
│  ┃  Command-line training script                                       ┃  │
│  ┃  ├─► Loads GR00T config                                            ┃  │
│  ┃  ├─► Creates OpenPI model config                                   ┃  │
│  ┃  ├─► Converts to OpenPI training config                            ┃  │
│  ┃  └─► Runs training                                                  ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                                              │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃ validate_groot_integration.py                                       ┃  │
│  ┃                                                                      ┃  │
│  ┃  5-step validation script                                           ┃  │
│  ┃  ├─► ✓ Validate imports                                            ┃  │
│  ┃  ├─► ✓ Load GR00T config                                           ┃  │
│  ┃  ├─► ✓ Convert to OpenPI config                                    ┃  │
│  ┃  ├─► ✓ Instantiate dataset                                         ┃  │
│  ┃  └─► ✓ Sample data                                                 ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                │ Produces
                                │
                    ┌───────────▼──────────┐
                    │  OpenPI TrainConfig  │
                    │                      │
                    │  With GR00T settings:│
                    │  ✓ action_horizon    │
                    │  ✓ action_dim        │
                    │  ✓ video_views       │
                    │  ✓ dataset_path      │
                    │  ✓ embodiment_tag    │
                    └───────────┬──────────┘
                                │
                                │ Used by
                                │
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPENPI TRAINING (EXISTING)                                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ create_torch_dataset(data_config, action_horizon, model_config)      │  │
│  │         │                                                             │  │
│  │         ├─► Sees gr00t_dataset_path is set                          │  │
│  │         └─► Creates Gr00tLeRobotTorchDataset                        │  │
│  │                     │                                                 │  │
│  │                     ▼                                                 │  │
│  │         ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │  │
│  │         ┃ Gr00tLeRobotTorchDataset (EXISTING ADAPTER)        ┃     │  │
│  │         ┃                                                     ┃     │  │
│  │         ┃  Uses GR00T's:                                     ┃     │  │
│  │         ┃  ├─► LeRobotEpisodeLoader                         ┃     │  │
│  │         ┃  ├─► StateActionProcessor                         ┃     │  │
│  │         ┃  ├─► MODALITY_CONFIGS                             ┃     │  │
│  │         ┃  └─► Action representations                        ┃     │  │
│  │         ┃                                                     ┃     │  │
│  │         ┃  Returns OpenPI format:                           ┃     │  │
│  │         ┃  {                                                 ┃     │  │
│  │         ┃    "observation.images.<view>": [H,W,3],         ┃     │  │
│  │         ┃    "observation.state": [state_dim],             ┃     │  │
│  │         ┃    "actions": [horizon, action_dim],             ┃     │  │
│  │         ┃    "prompt": str                                  ┃     │  │
│  │         ┃  }                                                 ┃     │  │
│  │         ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ OpenPI Transforms                                                     │  │
│  │ ├─► Repack (rename keys)                                            │  │
│  │ ├─► Resize images                                                    │  │
│  │ ├─► Normalize (using OpenPI or GR00T stats)                         │  │
│  │ └─► Tokenize text                                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Model Forward Pass                                                    │  │
│  │ ├─► Vision encoder (SigLIP)                                         │  │
│  │ ├─► Text encoder (Gemma)                                            │  │
│  │ ├─► Transformer                                                      │  │
│  │ └─► Action head                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Training Loop                                                         │  │
│  │ ├─► Compute loss                                                     │  │
│  │ ├─► Backprop                                                         │  │
│  │ ├─► Update weights                                                   │  │
│  │ └─► Log metrics                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Benefits

```
┌────────────────────────────────────────────────────────────────────┐
│                     BEFORE INTEGRATION                              │
│                                                                     │
│  OpenPI Training                        GR00T Training             │
│  ┌─────────────────┐                   ┌─────────────────┐        │
│  │ Manual config   │                   │ YAML config     │        │
│  │ - action_horizon│                   │ - Modality cfg  │        │
│  │ - action_dim    │                   │ - Auto settings │        │
│  │ - Manual setup  │                   │                 │        │
│  └─────────────────┘                   └─────────────────┘        │
│                                                                     │
│  Risk of mismatch!                      Automated setup            │
│  Hard to compare models                 Consistent training        │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     AFTER INTEGRATION                               │
│                                                                     │
│              ┌─────────────────────────────────┐                   │
│              │   Single Source of Truth        │                   │
│              │   (GR00T YAML + Modality Config)│                   │
│              └──────────────┬──────────────────┘                   │
│                             │                                       │
│                ┌────────────┴────────────┐                         │
│                │                         │                         │
│                ▼                         ▼                         │
│    ┌─────────────────┐       ┌─────────────────┐                  │
│    │ OpenPI Training │       │ GR00T Training  │                  │
│    │ - Same horizon  │       │ - Same horizon  │                  │
│    │ - Same dim      │       │ - Same dim      │                  │
│    │ - Same actions  │       │ - Same actions  │                  │
│    └─────────────────┘       └─────────────────┘                  │
│                                                                     │
│    ✅ Guaranteed consistency                                       │
│    ✅ Easy comparison                                              │
│    ✅ Single config to maintain                                    │
└────────────────────────────────────────────────────────────────────┘
```

## File Dependencies

```
gr00t_config_loader.py
    │
    ├─► Imports (from GR00T)
    │   ├─► gr00t.configs.base_config.Config
    │   ├─► gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS
    │   └─► gr00t.data.types (ActionConfig, ActionRepresentation, etc.)
    │
    ├─► Imports (from OpenPI)
    │   ├─► openpi.models.model
    │   ├─► openpi.training.config
    │   └─► openpi.transforms
    │
    └─► Exports
        ├─► load_gr00t_config()
        ├─► Gr00tConfigBundle
        └─► create_openpi_config_from_gr00t()

train_with_groot_config.py
    │
    ├─► Imports
    │   ├─► openpi.training.gr00t_config_loader.load_gr00t_config
    │   ├─► openpi.models (pi0, pi0_fast)
    │   └─► openpi.scripts.train.run_training
    │
    └─► CLI Entry Point
        └─► main() - tyro.cli interface

validate_groot_integration.py
    │
    ├─► Imports
    │   ├─► openpi.training.gr00t_config_loader
    │   ├─► openpi.training.data_loader
    │   └─► openpi.models.pi0
    │
    └─► CLI Entry Point
        └─► main() - tyro.cli interface
```

## Configuration Flow

```
GR00T YAML Config
    ├─► data.datasets[0].dataset_paths[0] ────────────────┐
    ├─► data.datasets[0].embodiment_tag ──────────────────┤
    ├─► data.video_backend ───────────────────────────────┤
    ├─► training.learning_rate ───────────────────────────┤
    ├─► training.warmup_steps ────────────────────────────┤
    ├─► training.max_steps ───────────────────────────────┤
    ├─► training.global_batch_size ───────────────────────┤
    └─► training.weight_decay ────────────────────────────┤
                                                            │
GR00T Modality Config                                      │
    ├─► modality_configs[tag]["action"].delta_indices ────┤
    ├─► modality_configs[tag]["action"].action_configs ───┤
    └─► modality_configs[tag]["observation"].modality_keys┤
                                                            │
                                                            │
                            ┌───────────────────────────────┘
                            │
                            ▼
                    Gr00tConfigBundle
                            │
                            ├─► get_action_horizon() = len(delta_indices)
                            ├─► get_action_dim() = sum(action dimensions)
                            └─► get_video_views() = observation keys
                                        │
                                        ▼
                                OpenPI TrainConfig
                                        │
                                        ├─► model.action_horizon
                                        ├─► model.action_dim
                                        ├─► data.gr00t_dataset_path
                                        ├─► data.gr00t_embodiment_tag
                                        ├─► data.gr00t_video_views
                                        ├─► lr_schedule.init_lr
                                        ├─► lr_schedule.warmup_steps
                                        ├─► num_train_steps
                                        ├─► batch_size
                                        └─► optimizer.weight_decay
```

## Action Representation Preservation

```
┌─────────────────────────────────────────────────────────────────┐
│                     GR00T Action Configs                         │
│                                                                  │
│  ActionConfig(                                                   │
│    rep=ActionRepresentation.RELATIVE,                          │
│    type=ActionType.EEF,                                        │
│    format=ActionFormat.XYZ_ROT6D,                              │
│    state_key="left_eef_pose",                                  │
│    normalization_type="percentile",                            │
│  )                                                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Stored in MODALITY_CONFIGS
                            │
            ┌───────────────▼────────────────┐
            │  GR00T modality_configs dict   │
            │  (loaded by import)            │
            └───────────────┬────────────────┘
                            │
                            │ Referenced by
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              Gr00tLeRobotTorchDataset                            │
│                                                                  │
│  __init__(spec):                                                │
│    self.embodiment_tag = spec.embodiment_tag                    │
│    self.modality_configs = MODALITY_CONFIGS[self.embodiment_tag]│
│    # Action configs automatically used                          │
│                                                                  │
│  __getitem__(idx):                                              │
│    raw_data = load_from_parquet(idx)                           │
│    # StateActionProcessor uses action_configs internally        │
│    processed = apply_action_representation(raw_data)           │
│    return openpi_format(processed)                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Returns
                            │
                ┌───────────▼──────────┐
                │  OpenPI Sample       │
                │                      │
                │  Actions are:        │
                │  ✓ RELATIVE applied  │
                │  ✓ Normalized        │
                │  ✓ Correct horizon   │
                │  ✓ Correct dimension │
                └──────────────────────┘
```

## Summary

This integration enables seamless training of OpenPI models using GR00T's data infrastructure with:

✅ **1 config loader** (`gr00t_config_loader.py`)  
✅ **1 training script** (`train_with_groot_config.py`)  
✅ **1 validation script** (`validate_groot_integration.py`)  
✅ **3 documentation files** (integration guide, README, summary)  
✅ **Zero modifications** to existing OpenPI or GR00T code  
✅ **Full preservation** of GR00T's action representations  
✅ **Easy usage** with command-line interface  

**Total implementation**: ~1,600 lines of code + ~1,600 lines of documentation
