#!/bin/bash

set -euo pipefail

OPENPI_ROOT_FSW="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/nigeln/JHU/11_openpi_surgical"
TEMPLATE_PATH="${OPENPI_ROOT_FSW}/eval_scripts/eval_template_1gpu.slurm"
CONFIG_DIR="${OPENPI_ROOT_FSW}/eval_scripts/generated"

CHECKPOINT_DIR="checkpoints/pi05_gr00t_local/exp04_lazy_row_load/25000"
TRAIN_CONFIG_NAME="pi05_gr00t_local"
EVAL_NAME="exp04_lazy_row_load_25000"
ACTION_HORIZON=""
DATASET_PATH=""
MODALITY_CONFIG_PATH=""
STATS_PATH=""
EMBODIMENT_TAG=""
VIDEO_VIEWS=""
EPISODE_IDS=""
OUTPUT_DIR=""
DEFAULT_PROMPT=""
STATS_KEY=""
SERVER_PORT="8000"
SERVER_TIMEOUT_MS="30000"
NUM_EPISODES=""
SAVE_PLOTS="false"
INFERENCE_STRIDE=""

usage() {
  echo "Usage: $0 --dataset-path PATH --modality-config PATH --stats-path PATH --embodiment-tag TAG --video-views \"v1 v2\" --episode-ids \"1,2,3\" --action-horizon N [options]"
  echo "Options:"
  echo "  --eval-name NAME"
  echo "  --checkpoint-dir PATH"
  echo "  --train-config NAME"
  echo "  --stats-key KEY"
  echo "  --output-dir PATH"
  echo "  --default-prompt TEXT"
  echo "  --server-port PORT"
  echo "  --server-timeout-ms MS"
  echo "  --num-episodes N"
  echo "  --save-plots true|false"
  echo "  --inference-stride N"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --modality-config)
      MODALITY_CONFIG_PATH="$2"; shift 2 ;;
    --stats-path)
      STATS_PATH="$2"; shift 2 ;;
    --embodiment-tag)
      EMBODIMENT_TAG="$2"; shift 2 ;;
    --video-views)
      VIDEO_VIEWS="$2"; shift 2 ;;
    --episode-ids)
      EPISODE_IDS="$2"; shift 2 ;;
    --action-horizon)
      ACTION_HORIZON="$2"; shift 2 ;;
    --eval-name)
      EVAL_NAME="$2"; shift 2 ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"; shift 2 ;;
    --train-config)
      TRAIN_CONFIG_NAME="$2"; shift 2 ;;
    --stats-key)
      STATS_KEY="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --default-prompt)
      DEFAULT_PROMPT="$2"; shift 2 ;;
    --server-port)
      SERVER_PORT="$2"; shift 2 ;;
    --server-timeout-ms)
      SERVER_TIMEOUT_MS="$2"; shift 2 ;;
    --num-episodes)
      NUM_EPISODES="$2"; shift 2 ;;
    --save-plots)
      SAVE_PLOTS="$2"; shift 2 ;;
    --inference-stride)
      INFERENCE_STRIDE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$DATASET_PATH" || -z "$MODALITY_CONFIG_PATH" || -z "$STATS_PATH" || -z "$EMBODIMENT_TAG" || -z "$VIDEO_VIEWS" || -z "$EPISODE_IDS" || -z "$ACTION_HORIZON" ]]; then
  echo "ERROR: Missing required arguments."
  usage
  exit 1
fi

if [[ -z "$STATS_KEY" ]]; then
  STATS_KEY="$(basename "$DATASET_PATH")"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="./rollout_results/${EVAL_NAME}"
fi

if [[ -z "$INFERENCE_STRIDE" ]]; then
  INFERENCE_STRIDE="${ACTION_HORIZON}"
fi

mkdir -p "$CONFIG_DIR"

IFS=',' read -r -a EPISODE_ID_ARRAY <<< "$EPISODE_IDS"
if [[ ${#EPISODE_ID_ARRAY[@]} -eq 1 ]]; then
  read -r -a EPISODE_ID_ARRAY <<< "$EPISODE_IDS"
fi

if [[ -z "$NUM_EPISODES" ]]; then
  NUM_EPISODES="${#EPISODE_ID_ARRAY[@]}"
fi

read -r -a VIDEO_VIEW_ARRAY <<< "$VIDEO_VIEWS"

CONFIG_PATH="${CONFIG_DIR}/${EVAL_NAME}.yaml"
cat > "$CONFIG_PATH" <<EOF
eval_name: "${EVAL_NAME}"
train_config_name: "${TRAIN_CONFIG_NAME}"
checkpoint_dir: "${CHECKPOINT_DIR}"
stats_path: "${STATS_PATH}"
stats_key: "${STATS_KEY}"
dataset_path: "${DATASET_PATH}"
embodiment_tag: "${EMBODIMENT_TAG}"
modality_config_path: "${MODALITY_CONFIG_PATH}"
video_views:
$(printf '  - "%s"\n' "${VIDEO_VIEW_ARRAY[@]}")
action_horizon: ${ACTION_HORIZON}
episode_ids:
$(printf '  - %s\n' "${EPISODE_ID_ARRAY[@]}")
num_episodes: ${NUM_EPISODES}
default_prompt: ${DEFAULT_PROMPT:+"${DEFAULT_PROMPT}"}
server_port: ${SERVER_PORT}
server_timeout_ms: ${SERVER_TIMEOUT_MS}
output_dir: "${OUTPUT_DIR}"
save_plots: ${SAVE_PLOTS}
inference_stride: ${INFERENCE_STRIDE}
EOF

echo "Wrote eval config: ${CONFIG_PATH}"

sbatch --export=ALL,EVAL_CONFIG_PATH="${CONFIG_PATH}" "${TEMPLATE_PATH}"
