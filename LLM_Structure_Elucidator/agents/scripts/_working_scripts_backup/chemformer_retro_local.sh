#!/bin/bash

# Base directories - use relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$SCRIPT_DIR/../.."
CONFIG_BASE_DIR="$BASE_DIR/.."

# Create necessary directories
TEMP_FOLDER="$BASE_DIR/_temp_folder"

# Default values for parameters
INPUT_FILE="$TEMP_FOLDER/retro_targets.txt"
OUTPUT_FILE="$TEMP_FOLDER/retro_predictions.csv"
MODEL_PATH="$CONFIG_BASE_DIR/chemformer_public/models/fined-tuned/uspto_50/last_v2.ckpt"
VOCAB_PATH="$CONFIG_BASE_DIR/chemformer_public/bart_vocab_downstream.json"
BATCH_SIZE=64
N_BEAMS=50
N_UNIQUE_BEAMS=None

# Initialize environment
# Note: These lines should be adjusted based on local environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate chemformer
else
    echo "Warning: conda not found, assuming environment is already set up"
fi

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
python "$BASE_DIR/agents/scripts/chemformer_retro_script.py" \
    --input_file="$INPUT_FILE" \
    --output_file="$OUTPUT_FILE" \
    --vocab_path="$VOCAB_PATH" \
    --model_path="$MODEL_PATH" \
    --batch_size="$BATCH_SIZE" \
    --n_beams="$N_BEAMS" \
    --n_unique_beams="$N_UNIQUE_BEAMS"
