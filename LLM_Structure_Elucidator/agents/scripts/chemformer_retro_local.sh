#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directories (relative to script location)
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels to LLM_Structure_Elucidator root
CONFIG_BASE_DIR="$(cd "$BASE_DIR/.." && pwd)"  # Go up one more level to find chemformer_public

# Create necessary directories
TEMP_FOLDER="$BASE_DIR/_temp_folder"
mkdir -p "$TEMP_FOLDER"
mkdir -p "$(dirname "$TEMP_FOLDER/retro_predictions.csv")"

# Log file for capturing output
LOG_FILE="$TEMP_FOLDER/chemformer_retro.log"

# Default values for parameters
INPUT_FILE="$TEMP_FOLDER/retro_targets.txt"
OUTPUT_FILE="$TEMP_FOLDER/retro_predictions.csv"
MODEL_PATH="$CONFIG_BASE_DIR/models/chemformer/fined-tuned/uspto_50/last_v2.ckpt"
VOCAB_PATH="$CONFIG_BASE_DIR/chemformer_public/bart_vocab_downstream.json"
BATCH_SIZE=64
N_BEAMS=20 #50
N_UNIQUE_BEAMS=-1
TEMPERATURE=1.0
SAMPLING_METHOD="beam"

# Initialize environment
# NOTE: Change these paths according to your CUDA and conda installation
module load CUDA/11.3.1  # Modify this according to your CUDA module system
eval "$(/projects/cc/se_users/knlr326/miniconda_SE/bin/conda shell.bash hook)"  # Change to your conda path
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/chemformer  # Change to your conda environment path

# Start the log file with a timestamp and configuration info
{
  echo "========================================================"
  echo "Chemformer Retrosynthesis Run - $(date)"
  echo "========================================================"
  echo "Python path: $(which python)"
  echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
  echo ""
  echo "Configuration:"
  echo "  Input file: $INPUT_FILE"
  echo "  Output file: $OUTPUT_FILE"
  echo "  Model path: $MODEL_PATH"
  echo "  Vocabulary path: $VOCAB_PATH"
  echo "  Batch size: $BATCH_SIZE"
  echo "  Number of beams: $N_BEAMS"
  echo "  Number of unique beams: $N_UNIQUE_BEAMS"
  echo "  Temperature: $TEMPERATURE"
  echo "  Sampling method: $SAMPLING_METHOD"
  echo ""
  echo "========================================================"
} > "$LOG_FILE"

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
        --temperature=*)
            TEMPERATURE="${1#*=}"
            shift
            ;;
        --sampling_method=*)
            SAMPLING_METHOD="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1" | tee -a "$LOG_FILE"
            exit 1
            ;;
    esac
done

# Ensure directories exist
mkdir -p "$(dirname "$INPUT_FILE")"

# Verify file existence
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at $MODEL_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$VOCAB_PATH" ]; then
    echo "Error: Vocabulary file not found at $VOCAB_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" >> "$LOG_FILE"
echo "Starting Chemformer prediction at $(date)" >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Run the Python script with unbuffered output and capture all output to the log file
# PYTHONUNBUFFERED=1 ensures Python doesn't buffer its output
PYTHONUNBUFFERED=1 python -u "$SCRIPT_DIR/chemformer_retro_script.py" \
    --input_file="$INPUT_FILE" \
    --output_file="$OUTPUT_FILE" \
    --vocab_path="$VOCAB_PATH" \
    --model_path="$MODEL_PATH" \
    --batch_size="$BATCH_SIZE" \
    --n_beams="$N_BEAMS" \
    --n_unique_beams="$N_UNIQUE_BEAMS" \
    --temperature="$TEMPERATURE" \
    --sampling_method="$SAMPLING_METHOD" 2>&1 | tee -a "$LOG_FILE"

RETURN_CODE=${PIPESTATUS[0]}

{
  echo ""
  echo "========================================================"
  echo "Chemformer prediction completed at $(date) with return code: $RETURN_CODE"
  
  if [ $RETURN_CODE -eq 0 ]; then
    echo "Results saved to: $OUTPUT_FILE"
    
    # Display a sample of the results if the file exists
    if [ -f "$OUTPUT_FILE" ]; then
      echo ""
      echo "Sample of results (first 5 lines):"
      head -n 5 "$OUTPUT_FILE"
    fi
  else
    echo "ERROR: Chemformer prediction failed with return code $RETURN_CODE"
  fi
  
  echo "========================================================"
} | tee -a "$LOG_FILE"

echo "Log file saved to: $LOG_FILE" | tee -a "$LOG_FILE"