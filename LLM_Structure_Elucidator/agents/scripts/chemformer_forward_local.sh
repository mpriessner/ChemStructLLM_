#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directories (relative to script location)
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels to LLM_Structure_Elucidator root
CONFIG_BASE_DIR="$(cd "$BASE_DIR/.." && pwd)"  # Go up one more level to find chemformer_public

# Create necessary directories
TEMP_FOLDER="$BASE_DIR/_temp_folder"
mkdir -p "$TEMP_FOLDER"
mkdir -p "$(dirname "$TEMP_FOLDER/forward_predictions.csv")"

# Default values for parameters
INPUT_FILE="$TEMP_FOLDER/forward_targets.txt"
OUTPUT_FILE="$TEMP_FOLDER/forward_predictions.csv"
MODEL_PATH="$CONFIG_BASE_DIR/models/chemformer/fined-tuned/uspto_50/last_v2.ckpt"
VOCAB_PATH="$CONFIG_BASE_DIR/chemformer_public/bart_vocab_downstream.json"
BATCH_SIZE=64
N_BEAMS=20 #50
N_UNIQUE_BEAMS=-1

# Initialize environment
# NOTE: Change these paths according to your CUDA and conda installation
module load CUDA/11.3.1  # Modify this according to your CUDA module system
eval "$(/projects/cc/se_users/knlr326/miniconda_SE/bin/conda shell.bash hook)"  # Change to your conda path
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/chemformer  # Change to your conda environment path

# Verify environment
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file=*)
            INPUT_FILE="${1#*=}"
            shift
            ;;
        --output_file=*)
            OUTPUT_FILE="${1#*=}"
            shift
            ;;
        --model_path=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --vocab_path=*)
            VOCAB_PATH="${1#*=}"
            shift
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --n_beams=*)
            N_BEAMS="${1#*=}"
            shift
            ;;
        --n_unique_beams=*)
            N_UNIQUE_BEAMS="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Ensure directories exist
mkdir -p "$(dirname "$INPUT_FILE")"

# Verify file existence
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$VOCAB_PATH" ]; then
    echo "Error: Vocabulary file not found at $VOCAB_PATH"
    exit 1
fi

# Run the Python script
python "$BASE_DIR/agents/scripts/chemformer_forward_script.py" \
    --input_file="$INPUT_FILE" \
    --output_file="$OUTPUT_FILE" \
    --vocab_path="$VOCAB_PATH" \
    --model_path="$MODEL_PATH" \
    --batch_size="$BATCH_SIZE" \
    --n_beams="$N_BEAMS" \
    --n_unique_beams="$N_UNIQUE_BEAMS"
