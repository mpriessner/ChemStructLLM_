#!/bin/bash

# Base directories
BASE_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator"
CONFIG_BASE_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability"

# Create necessary directories for logging
TEMP_FOLDER="$BASE_DIR/_temp_folder/mmst_temp"
mkdir -p "$TEMP_FOLDER"

# Create a timestamped run directory if not provided
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Initialize variables
RUN_DIR=""
INPUT_CSV=""
MODEL_SAVE_DIR=""
CONFIG_DIR="$CONFIG_BASE_DIR/utils_MMT"
RUN_MODE="both"
IC_THRESHOLD=0.2
EXP_DATA_PATH="$BASE_DIR/data/molecular_data/molecular_data.json"
MULTINOM_RUNS=50 #30
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
IMPROVEMENT_CYCLES=1 # 3

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
        --mol2mol_model_path=*)
            MOL2MOL_MODEL_PATH="${1#*=}"
            shift
            ;;
        --mol2mol_vocab_path=*)
            MOL2MOL_VOCAB_PATH="${1#*=}"
            shift
            ;;
        --MF_delta_weight=*)
            MF_DELTA_WEIGHT="${1#*=}"
            shift
            ;;
        --tanimoto_filter=*)
            TANIMOTO_FILTER="${1#*=}"
            shift
            ;;
        --MF_max_trails=*)
            MF_MAX_TRAILS="${1#*=}"
            shift
            ;;
        --max_scaffold_generations=*)
            MAX_SCAFFOLD_GENERATIONS="${1#*=}"
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
RUN_DIR=${RUN_DIR:-"$TEMP_FOLDER/run_$TIMESTAMP"}
INPUT_CSV=${INPUT_CSV:-"$RUN_DIR/mmst_input.csv"}
SGNN_GEN_FOLDER=${SGNN_GEN_FOLDER:-"$RUN_DIR/sgnn_output"}
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"$RUN_DIR/models"}

# Create log file
LOG_FILE="$RUN_DIR/mmst_execution_$(date +"%Y%m%d_%H%M%S").log"
mkdir -p "$(dirname "$LOG_FILE")"

# Start the log file with a timestamp and configuration info
{
  echo "========================================================"
  echo "MMST Structure Prediction Run - $(date)"
  echo "========================================================"
  echo "Run Directory: $RUN_DIR"
  echo ""
} > "$LOG_FILE"

# Initialize environment
echo "ACTIVATE" | tee -a "$LOG_FILE"
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator
module load CUDA/11.3.1

# Set wandb directory
export WANDB_DIR="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/wandb"
mkdir -p "$WANDB_DIR"

# Add debug mode
set -x

{
  echo "Script started at $(date)"
  echo "Current directory: $(pwd)"
  echo "Python path: $(which python)"
  echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
  echo ""
  echo "Configuration:"
  echo "  Input CSV: $INPUT_CSV"
  echo "  Output directory: $RUN_DIR"
  echo "  Model save directory: $MODEL_SAVE_DIR"
  echo "  Config directory: $CONFIG_DIR"
  echo "  Mol2Mol model path: $MOL2MOL_MODEL_PATH"
  echo "  Mol2Mol vocabulary path: $MOL2MOL_VOCAB_PATH"
  echo "  SGNN generation folder: $SGNN_GEN_FOLDER"
  echo "  MF delta weight: $MF_DELTA_WEIGHT"
  echo "  Tanimoto filter: $TANIMOTO_FILTER"
  echo "  MF max trails: $MF_MAX_TRAILS"
  echo "  Max scaffold generations: $MAX_SCAFFOLD_GENERATIONS"
  echo "  MF generations: $MF_GENERATIONS"
  echo "  Number of epochs: $NUM_EPOCHS"
  echo "  Prediction batch size: $PREDICTION_BATCH_SIZE"
  echo "  Improvement cycles: $IMPROVEMENT_CYCLES"
  echo "  IC threshold: $IC_THRESHOLD"
  echo "  Learning rate: $LEARNING_RATE"
  echo "  MW tolerance: $MW_TOLERANCE"
  echo "  NMR types: ${NMR_TYPES[*]}"
  echo ""
  echo "========================================================"
} | tee -a "$LOG_FILE"

# Validate required parameters
echo "Validating parameters..." | tee -a "$LOG_FILE"
if [ -z "$INPUT_CSV" ] || [ -z "$RUN_DIR" ] || [ -z "$CONFIG_DIR" ]; then
    echo "Error: Required parameters missing" | tee -a "$LOG_FILE"
    echo "Usage: $0 --input_csv=<path> --run_dir=<path> --config_dir=<path> [options]" | tee -a "$LOG_FILE"
    exit 1
fi

# Verify file existence
if [ ! -f "$MOL2MOL_MODEL_PATH" ]; then
    echo "Error: Mol2Mol model checkpoint not found at $MOL2MOL_MODEL_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$MOL2MOL_VOCAB_PATH" ]; then
    echo "Error: Mol2Mol vocabulary file not found at $MOL2MOL_VOCAB_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

# Create required directories
echo "Creating directories..." | tee -a "$LOG_FILE"
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

echo "" | tee -a "$LOG_FILE"
echo "Starting MMST structure prediction at $(date)" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run MMST script with unbuffered output
PYTHONUNBUFFERED=1 python -u "$SCRIPT_DIR/mmst_script.py" \
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
    --nmr_types ${NMR_TYPES[@]} 2>&1 | tee -a "$LOG_FILE"

RETURN_CODE=${PIPESTATUS[0]}

{
  echo ""
  echo "========================================================"
  echo "MMST structure prediction completed at $(date) with return code: $RETURN_CODE"
  
  if [ $RETURN_CODE -eq 0 ]; then
    echo "Results saved to: $RUN_DIR"
    
    # If results_dict.json exists, display summary of molecules
    RESULTS_FILE="$RUN_DIR/results_dict.json"
    if [ -f "$RESULTS_FILE" ]; then
      echo ""
      echo "Generated molecules summary:"
      # Extract source molecules (keys) and generated SMILES (first element of each inner list)
      python -c "
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    
    print(f'Total source molecules: {len(results.keys())}')
    total_generated = 0
    
    for source, generations in results.items():
        gen_count = len(generations)
        total_generated += gen_count
        print(f'Source: {source[:30]}... -> {gen_count} molecules generated')
        
        # Show first 2 generated molecules for each source
        for i, gen in enumerate(generations[:2]):
            if isinstance(gen, list) and len(gen) > 0:
                print(f'  {i+1}. {gen[0][:50]}...')
    
    print(f'Total generated molecules: {total_generated}')
except Exception as e:
    print(f'Error processing results: {e}')
" 
    fi
    
    echo "MMST structure prediction completed successfully"
  else
    echo "ERROR: MMST structure prediction failed with return code $RETURN_CODE"
  fi
  
  echo "========================================================"
  echo "Log file saved to: $LOG_FILE"
} | tee -a "$LOG_FILE"

# Return the exit code from the Python script
exit $RETURN_CODE
