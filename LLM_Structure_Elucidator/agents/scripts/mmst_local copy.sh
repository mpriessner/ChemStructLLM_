#!/bin/bash

# Base directories
BASE_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator"
CONFIG_BASE_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability"

# Initialize variables
RUN_DIR=""
INPUT_CSV=""
MODEL_SAVE_DIR=""
CONFIG_DIR="$CONFIG_BASE_DIR/utils_MMT"
RUN_MODE="both"
IC_THRESHOLD=0.2
EXP_DATA_PATH="$BASE_DIR/data/molecular_data/molecular_data.json"
MULTINOM_RUNS=30 #30
LEARNING_RATE=0.00005 #0.0002
MW_TOLERANCE=0.5  

# Mol2Mol parameters
MOL2MOL_MODEL_PATH="$CONFIG_BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/weights_pubchem_with_counts_and_rank_sanitized.ckpt"
MOL2MOL_VOCAB_PATH="$CONFIG_BASE_DIR/deep-molecular-optimization/experiments/trained/Alessandro_big/vocab_new.pkl"
MF_DELTA_WEIGHT=50
TANIMOTO_FILTER=0.7
MF_MAX_TRAILS=1000 # 500
MAX_SCAFFOLD_GENERATIONS=200 # 200
 
# SGNN parameters
SGNN_GEN_FOLDER=""
declare -a NMR_TYPES=("1H" "13C" "COSY" "HSQC")
# declare -a NMR_TYPES=("HSQC")

# MMST parameters
MF_GENERATIONS=10 # 200
NUM_EPOCHS=10 # 15
PREDICTION_BATCH_SIZE=32
IMPROVEMENT_CYCLES=3 # 3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_dir=*)
            RUN_DIR="${1#*=}"
            shift
            ;;
        --input_csv=*)
            INPUT_CSV="${1#*=}"
            shift
            ;;
        --model_save_dir=*)
            MODEL_SAVE_DIR="${1#*=}"
            shift
            ;;
        --mw_tolerance=*)  
            MW_TOLERANCE="${1#*=}"
            shift
            ;;
        --config_dir=*)
            CONFIG_DIR="${1#*=}"
            shift
            ;;
        --sgnn_gen_folder=*)
            SGNN_GEN_FOLDER="${1#*=}"
            shift
            ;;
        --run_mode=*)
            RUN_MODE="${1#*=}"
            shift
            ;;
        --multinom_runs=*)
            MULTINOM_RUNS="${1#*=}"
            shift
            ;;
        --learning_rate=*)
            LEARNING_RATE="${1#*=}"
            shift
            ;;
        --MF_generations=*)
            MF_GENERATIONS="${1#*=}"
            shift
            ;;
        --num_epochs=*)
            NUM_EPOCHS="${1#*=}"
            shift
            ;;
        --prediction_batch_size=*)
            PREDICTION_BATCH_SIZE="${1#*=}"
            shift
            ;;
        --improvement_cycles=*)
            IMPROVEMENT_CYCLES="${1#*=}"
            shift
            ;;
        --IC_threshold=*)
            IC_THRESHOLD="${1#*=}"
            shift
            ;;
        --exp_data_path=*)
            EXP_DATA_PATH="${1#*=}"
            shift
            ;;
        --nmr_types=*)
            IFS=',' read -ra NMR_TYPES <<< "${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

# Set default values if not provided
RUN_DIR=${RUN_DIR:-"$BASE_DIR/_temp_folder/mmst_temp"}
INPUT_CSV=${INPUT_CSV:-"$RUN_DIR/mmst_input.csv"}
SGNN_GEN_FOLDER=${SGNN_GEN_FOLDER:-"$RUN_DIR/sgnn_output"}
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"$RUN_DIR/models"}

echo "ACTIVATE"
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator
module load CUDA/11.3.1

# Set wandb directory
export WANDB_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/wandb"
mkdir -p "$WANDB_DIR"

# Add debug mode
set -x

echo "Script started at $(date)"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"

# Validate required parameters
echo "Validating parameters..."
if [ -z "$INPUT_CSV" ] || [ -z "$RUN_DIR" ] || [ -z "$CONFIG_DIR" ]; then
    echo "Error: Required parameters missing"
    echo "Usage: $0 --input_csv=<path> --run_dir=<path> --config_dir=<path> [options]"
    exit 1
fi

# Create required directories
echo "Creating directories..."
mkdir -p "$RUN_DIR" "$SGNN_GEN_FOLDER" "$MODEL_SAVE_DIR"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export all environment variables that mmst_script.py might need
export PYTHONPATH="$BASE_DIR:$CONFIG_BASE_DIR:$CONFIG_BASE_DIR/deep-molecular-optimization:$PYTHONPATH"
export MF_generations=$MF_GENERATIONS
export num_epochs=$NUM_EPOCHS
export prediction_batch_size=$PREDICTION_BATCH_SIZE
export improvement_cycles=$IMPROVEMENT_CYCLES
export IC_threshold=$IC_THRESHOLD
export learning_rate=$LEARNING_RATE
export MF_delta_weight=$MF_DELTA_WEIGHT
export tanimoto_filter=$TANIMOTO_FILTER ### need to change in config file
export MF_max_trails=$MF_MAX_TRAILS
export max_scaffold_generations=$MAX_SCAFFOLD_GENERATIONS

# Run MMST script
python "$SCRIPT_DIR/mmst_script.py" \
    --input_csv "$INPUT_CSV" \
    --output_dir "$RUN_DIR" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --config_dir "$CONFIG_DIR" \
    --mol2mol_model_path "$MOL2MOL_MODEL_PATH" \
    --mol2mol_vocab_path "$MOL2MOL_VOCAB_PATH" \
    --MF_delta_weight "$MF_DELTA_WEIGHT" \
    --tanimoto_filter "$TANIMOTO_FILTER" \
    --MF_max_trails "$MF_MAX_TRAILS" \
    --max_scaffold_generations "$MAX_SCAFFOLD_GENERATIONS" \
    --sgnn_gen_folder "$SGNN_GEN_FOLDER" \
    --MF_generations "$MF_GENERATIONS" \
    --num_epochs "$NUM_EPOCHS" \
    --prediction_batch_size "$PREDICTION_BATCH_SIZE" \
    --improvement_cycles "$IMPROVEMENT_CYCLES" \
    --IC_threshold "$IC_THRESHOLD" \
    --exp_data_path "$EXP_DATA_PATH" \
    --multinom_runs "$MULTINOM_RUNS" \
    --learning_rate "$LEARNING_RATE" \
    --mw_tolerance "$MW_TOLERANCE" \
    --nmr_types ${NMR_TYPES[@]}

# Check execution status
if [ $? -eq 0 ]; then
    echo "MMST structure prediction completed successfully"
    exit 0
else
    echo "Error: MMST structure prediction failed"
    exit 1
fi
