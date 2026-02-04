#!/bin/bash

# Memory-Optimized Training Script for OpenPI + GR00T
# This script provides several memory optimization modes for training

set -e

# Default values
MODE="balanced"
DATASET_PATH=""
MODALITY_CONFIG_PATH=""
EMBODIMENT_TAG="dvrk"
EXP_NAME=""
BATCH_SIZE=""
NUM_WORKERS=""
ADDITIONAL_ARGS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Memory-optimized training for OpenPI with GR00T dataset"
    echo ""
    echo "Required Options:"
    echo "  --dataset-path PATH          Path to GR00T dataset"
    echo "  --modality-config-path PATH  Path to modality config file"
    echo ""
    echo "Optional Options:"
    echo "  --mode MODE                  Memory optimization mode: ultra|balanced|custom (default: balanced)"
    echo "  --embodiment-tag TAG         Embodiment tag (default: dvrk)"
    echo "  --exp-name NAME              Experiment name (auto-generated if not provided)"
    echo "  --batch-size SIZE            Batch size (custom mode only)"
    echo "  --num-workers NUM            Number of workers (custom mode only)"
    echo "  --single-view                Use single camera view only (custom mode)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Memory Modes:"
    echo "  ultra    - Minimal memory usage (~12GB): batch_size=1, single view"
    echo "  balanced - Good balance (~22GB): batch_size=4, all views with compression"
    echo "  custom   - Custom settings: specify --batch-size and other options"
    echo ""
    echo "Examples:"
    echo "  # Ultra low memory mode"
    echo "  $0 --mode ultra --dataset-path /data/dataset --modality-config-path /path/config.py"
    echo ""
    echo "  # Balanced mode with custom experiment name"
    echo "  $0 --mode balanced --dataset-path /data/dataset --modality-config-path /path/config.py --exp-name my_experiment"
    echo ""
    echo "  # Custom mode with specific settings"
    echo "  $0 --mode custom --dataset-path /data/dataset --modality-config-path /path/config.py --batch-size 2 --single-view"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --modality-config-path)
            MODALITY_CONFIG_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --embodiment-tag)
            EMBODIMENT_TAG="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --single-view)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --single-view"
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET_PATH" ]]; then
    echo -e "${RED}Error: --dataset-path is required${NC}"
    print_usage
    exit 1
fi

if [[ -z "$MODALITY_CONFIG_PATH" ]]; then
    echo -e "${RED}Error: --modality-config-path is required${NC}"
    print_usage
    exit 1
fi

# Validate paths
if [[ ! -d "$DATASET_PATH" ]]; then
    echo -e "${RED}Error: Dataset path does not exist: $DATASET_PATH${NC}"
    exit 1
fi

if [[ ! -f "$MODALITY_CONFIG_PATH" ]]; then
    echo -e "${RED}Error: Modality config file does not exist: $MODALITY_CONFIG_PATH${NC}"
    exit 1
fi

# Validate mode
if [[ ! "$MODE" =~ ^(ultra|balanced|custom)$ ]]; then
    echo -e "${RED}Error: Invalid mode '$MODE'. Must be: ultra, balanced, or custom${NC}"
    exit 1
fi

# Generate experiment name if not provided
if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="${MODE}_memory_opt_$(date +%Y%m%d_%H%M%S)"
fi

# Add custom mode arguments
if [[ "$MODE" == "custom" ]]; then
    if [[ -n "$BATCH_SIZE" ]]; then
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS --batch-size $BATCH_SIZE"
    fi
    if [[ -n "$NUM_WORKERS" ]]; then
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS --num-workers $NUM_WORKERS"
    fi
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}MEMORY-OPTIMIZED TRAINING FOR OPENPI + GR00T${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Mode: $MODE"
echo "  Dataset: $DATASET_PATH"
echo "  Modality Config: $MODALITY_CONFIG_PATH"
echo "  Embodiment: $EMBODIMENT_TAG"
echo "  Experiment: $EXP_NAME"

# Show mode-specific information
case $MODE in
    ultra)
        echo -e "${YELLOW}Ultra Low Memory Mode:${NC}"
        echo "  - Batch size: 1"
        echo "  - Single camera view (endoscope_left)"
        echo "  - Action horizon: 32"
        echo "  - Video compression: enabled"
        echo "  - Expected memory: ~12GB per GPU"
        ;;
    balanced)
        echo -e "${YELLOW}Balanced Memory Mode:${NC}"
        echo "  - Batch size: 4"
        echo "  - All camera views with compression"
        echo "  - Action horizon: 32"
        echo "  - Video compression: enabled"
        echo "  - Expected memory: ~22GB per GPU"
        ;;
    custom)
        echo -e "${YELLOW}Custom Mode:${NC}"
        echo "  - Custom settings as specified"
        if [[ -n "$BATCH_SIZE" ]]; then
            echo "  - Batch size: $BATCH_SIZE"
        fi
        if [[ -n "$NUM_WORKERS" ]]; then
            echo "  - Workers: $NUM_WORKERS"
        fi
        ;;
esac

echo ""

# Check if memory optimization script exists
if [[ ! -f "train_memory_optimized.py" ]]; then
    echo -e "${RED}Error: train_memory_optimized.py not found in current directory${NC}"
    echo "Please make sure you're running this script from the OpenPI root directory"
    exit 1
fi

# Check if Python environment is set up
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Confirm before starting
echo -e "${YELLOW}Ready to start training. This will:${NC}"
echo "1. Set up memory-optimized environment variables"
echo "2. Configure video compression and caching"
echo "3. Start training with reduced memory usage"
echo ""

read -p "Continue? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo -e "${GREEN}Starting memory-optimized training...${NC}"
echo ""

# Set up environment variables for memory optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Build the command
CMD="python train_memory_optimized.py"
CMD="$CMD --mode $MODE"
CMD="$CMD --dataset-path \"$DATASET_PATH\""
CMD="$CMD --modality-config-path \"$MODALITY_CONFIG_PATH\""
CMD="$CMD --embodiment-tag $EMBODIMENT_TAG"
CMD="$CMD --exp-name $EXP_NAME"

if [[ -n "$ADDITIONAL_ARGS" ]]; then
    CMD="$CMD $ADDITIONAL_ARGS"
fi

echo -e "${BLUE}Executing:${NC} $CMD"
echo ""

# Run the training
eval $CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}TRAINING COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "Experiment: $EXP_NAME"
    echo "Check the logs and checkpoints for results."
else
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}TRAINING FAILED${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    echo "Check the error messages above for troubleshooting."
    echo "You may need to:"
    echo "1. Reduce batch size further (try --mode ultra)"
    echo "2. Use single camera view (add --single-view for custom mode)"
    echo "3. Check GPU memory with: nvidia-smi"
    exit 1
fi