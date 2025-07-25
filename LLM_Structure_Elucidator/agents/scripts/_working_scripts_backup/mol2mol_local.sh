#!/bin/bash

# Base directories

BASE_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator"
CONFIG_BASE_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability"


# Default values for parameters
INPUT_CSV="$BASE_DIR/_temp_folder/mol2mol_selection.csv"
CONFIG_DIR="$CONFIG_BASE_DIR"
OUTPUT_DIR="$BASE_DIR/_temp_folder"
MODEL_PATH="$CONFIG_BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/weights_pubchem_with_counts_and_rank_sanitized.ckpt"
VOCAB_PATH="$CONFIG_BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/vocab_new.pkl"

DELTA_WEIGHT=30
TANIMOTO_FILTER=0.2
NUM_GENERATIONS=100
MAX_TRIALS=100
MAX_SCAFFOLD_GENERATIONS=10

echo "ACTIVATE";
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate  /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator 
module load CUDA/11.3.1
echo $(which python)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_csv=*)
            INPUT_CSV="${1#*=}"
            shift
            ;;
        --output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --config_dir=*)
            CONFIG_DIR="${1#*=}"
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
        --delta_weight=*)
            # Convert float to integer by removing decimal part
            DELTA_WEIGHT=$(echo "${1#*=}" | cut -d. -f1)
            shift
            ;;
        --tanimoto_filter=*)
            TANIMOTO_FILTER="${1#*=}"
            shift
            ;;
        --num_generations=*)
            NUM_GENERATIONS="${1#*=}"
            shift
            ;;
        --max_trials=*)
            MAX_TRIALS="${1#*=}"
            shift
            ;;
        --max_scaffold_generations=*)
            MAX_SCAFFOLD_GENERATIONS="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$INPUT_CSV" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$CONFIG_DIR" ] || [ -z "$MODEL_PATH" ] || [ -z "$VOCAB_PATH" ]; then
    echo "Error: Required parameters missing"
    echo "Usage: $0 --input_csv=<path> --output_dir=<path> --config_dir=<path> [options]"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up Python environment (adjust as needed)
# Assuming you have a conda environment named 'mol2mol'
# conda activate mol2mol

# Run the Mol2Mol generation script
echo "Starting Mol2Mol generation..."
echo "Using parameters:"
echo "Input CSV: $INPUT_CSV"
echo "Output Directory: $OUTPUT_DIR"
echo "Config Directory: $CONFIG_DIR"
echo "Model Path: $MODEL_PATH"
echo "Vocabulary Path: $VOCAB_PATH"

PYTHON_CMD=(
    python "$SCRIPT_DIR/Mol2Mol_script.py"
    "--input_csv" "$INPUT_CSV"
    "--output_dir" "$OUTPUT_DIR"
    "--config_dir" "$CONFIG_DIR"
    "--model_path" "$MODEL_PATH"
    "--vocab_path" "$VOCAB_PATH"
    "--delta_weight" "$DELTA_WEIGHT"
    "--tanimoto_filter" "$TANIMOTO_FILTER"
    "--num_generations" "$NUM_GENERATIONS"
    "--max_trials" "$MAX_TRIALS"
    "--max_scaffold_generations" "$MAX_SCAFFOLD_GENERATIONS"
)

echo "Running command: ${PYTHON_CMD[*]}"
"${PYTHON_CMD[@]}"

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo "Mol2Mol generation completed successfully"
    exit 0
else
    echo "Error: Mol2Mol generation failed"
    exit 1
fi
