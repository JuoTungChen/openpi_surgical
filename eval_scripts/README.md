# Open-Loop Eval (SLURM)

This directory contains a SLURM launcher and template for open-loop evaluation
of OpenPI GR00T checkpoints with dataset-driven rollouts.

## Run (recommended)

```bash
bash eval_scripts/launch_open_loop_eval_1gpu.sh \
  --dataset-path /lustre/fsw/portfolios/healthcareeng/users/nigeln/cache/huggingface/lerobot/hf_jhu/hf_suturebot \
  --modality-config /lustre/fsw/portfolios/healthcareeng/users/nigeln/JHU/8_groot_n1.6/examples/dVRK/dVRK_config.py \
  --stats-path /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/nigeln/JHU/11_openpi_surgical/assets/percentile_stats_hf_suturebot.json \
  --stats-key hf_suturebot \
  --embodiment-tag dvrk \
  --video-views "endoscope_left wrist_left wrist_right" \
  --episode-ids "0,3,7,10" \
  --action-horizon 50 \
  --checkpoint-dir checkpoints/pi05_gr00t_local/exp04_lazy_row_load/25000 \
  --train-config pi05_gr00t_local \
  --save-plots true \
  --inference-stride 16
```

Notes:
- `--inference-stride` controls how often inference runs. If omitted, it defaults to `action_horizon`.
- Plots are saved under `rollout_results/<eval_name>/plots/` when `--save-plots true`.
- The launcher writes the YAML config to `eval_scripts/generated/<eval_name>.yaml` and submits via `sbatch`.

## Outputs

- Metrics: `rollout_results/<eval_name>/rollout_results.json`
- Plots (optional): `rollout_results/<eval_name>/plots/episode_<id>_actions.png`

## Troubleshooting

- If the job disappears from `squeue`, check both `logs/<job_id>/openpi_eval.out` and
  `logs/<job_id>/openpi_eval.err`.
