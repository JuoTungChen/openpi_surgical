#### load a local LeRobot dataset via `gr00t` (reuse embodiment configs)

If you have a **local LeRobot-format dataset** (with `meta/` and `data/`) and want to reuse `gr00t`'s data loader
and embodiment modality configs, `openpi` can import and use `gr00t` directly (no code copying).

- Install `gr00t` into the same Python environment as `openpi`:

```bash
pip install -e <PATH_TO_GR00TN1.6>
```

Or you can simply set the python path so the script can find where groot is.
For example, append this before uv run any file:
```
PYTHONPATH=<root>/openpi/src:<root>/gr00t_n1.6
```

- Use the example config `pi05_gr00t_local` in [`src/openpi/training/config.py`](src/openpi/training/config.py) and edit:
  - `dataset_path`: your local dataset root
  - `embodiment_tag`: a key from `gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS`
  - `repack_transforms`: map your dataset’s view names into the model’s expected image keys