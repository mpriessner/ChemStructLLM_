#!/bin/bash
#SBATCH --job-name=run_mol2mol_generation
#SBATCH --partition=short-gpu
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --time=0-1:00:00
# NOTE: Change this output path according to your setup
#SBATCH --output=/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/logs/mol2mol_sbatch_%j_%N.txt

#SBATCH --constraint=volta

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directories (relative to script location)
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels to LLM_Structure_Elucidator root
CONFIG_BASE_DIR="$(cd "$BASE_DIR/.." && pwd)"  # Go up one more level to find deep-molecular-optimization

# Default values for parameters
INPUT_CSV="$BASE_DIR/_temp_folder/mol2mol_selection.csv"
CONFIG_DIR="$CONFIG_BASE_DIR"
OUTPUT_DIR="$BASE_DIR/_temp_folder"
MODEL_PATH="$CONFIG_BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/weights_pubchem_with_counts_and_rank_sanitized.ckpt"
VOCAB_PATH="$CONFIG_BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/vocab_new.pkl"
DELTA_WEIGHT=30
TANIMOTO_FILTER=0.2
NUM_GENERATIONS=5 #100
MAX_TRIALS=100
MAX_SCAFFOLD_GENERATIONS=10

 
echo "ACTIVATE";
# NOTE: Change these paths according to your conda installation and environment
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate  /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator  # Change to your conda environment path
module load CUDA/11.3.1  # Modify this according to your CUDA module system
echo $(which python)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_csv=*|--input_csv)
            INPUT_CSV="${1#*=}"
            shift
            ;;
        --output_dir=*|--output_dir)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --config_dir=*|--config_dir)
            CONFIG_DIR="${1#*=}"
            shift
            ;;
        --model_path=*|--model_path)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --vocab_path=*|--vocab_path)
            VOCAB_PATH="${1#*=}"
            shift
            ;;
        --delta_weight=*|--delta_weight)
            # Convert float to integer by removing decimal part
            DELTA_WEIGHT=$(echo "${1#*=}" | cut -d. -f1)
            shift
            ;;
        --tanimoto_filter=*|--tanimoto_filter)
            TANIMOTO_FILTER="${1#*=}"
            shift
            ;;
        --num_generations=*|--num_generations)
            NUM_GENERATIONS="${1#*=}"
            shift
            ;;
        --max_trials=*|--max_trials)
            MAX_TRIALS="${1#*=}"
            shift
            ;;
        --max_scaffold_generations=*|--max_scaffold_generations)
            MAX_SCAFFOLD_GENERATIONS="${1#*=}"
            shift
            ;;
        *|--*)
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


echo "Starting Mol2Mol generation..."
echo "Using parameters:"
echo "Input CSV: $INPUT_CSV"
echo "Output Directory: $OUTPUT_DIR"
echo "Config Directory: $BASE_DIR"
echo "Model Path: $BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/weights_pubchem_with_counts_and_rank_sanitized.ckpt"
echo "Vocabulary Path: $BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/vocab_new.pkl"

# Debug: Check if script exists
SCRIPT_PATH="$SCRIPT_DIR/Mol2Mol_script.py"
echo "Checking script at: $SCRIPT_PATH"
echo "Listing all files in scripts directory with permissions:"
ls -la "$SCRIPT_DIR"
echo ""

if [ -f "$SCRIPT_PATH" ]; then
    echo "Script file exists"
    ls -l "$SCRIPT_PATH"
    echo "File permissions in octal:"
    stat -c "%a %n" "$SCRIPT_PATH"
else
    echo "Error: Script file not found"
    echo "Current directory: $(pwd)"
    echo "Script directory path: $SCRIPT_DIR"
    exit 1
fi

# Execute the Python script with parameters
CMD="python $SCRIPT_PATH \
    --input_csv $INPUT_CSV \
    --output_dir $OUTPUT_DIR \
    --config_dir $CONFIG_BASE_DIR \
    --model_path $MODEL_PATH \
    --vocab_path $VOCAB_PATH \
    --delta_weight $DELTA_WEIGHT \
    --tanimoto_filter $TANIMOTO_FILTER \
    --num_generations $NUM_GENERATIONS \
    --max_trials $MAX_TRIALS \
    --max_scaffold_generations $MAX_SCAFFOLD_GENERATIONS"

echo "Running command: $CMD"
eval $CMD

if [ $? -eq 0 ]; then
    echo "Mol2Mol generation completed successfully"
    exit 0
else
    echo "Error: Mol2Mol generation failed"
    exit 1
fi
