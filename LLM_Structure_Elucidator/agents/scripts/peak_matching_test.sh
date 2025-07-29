#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directories (relative to script location)
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels to LLM_Structure_Elucidator root
PROJECT_ROOT="$(cd "$BASE_DIR/.." && pwd)"  # Go up one more level to parent directory
INPUT_JSON="${BASE_DIR}/_temp_folder/peak_matching/current_run/input_data.json"

# HPC-specific setup - following SGNN pattern exactly
echo "python1"    

module purge
echo "python2"    
# NOTE: Change this according to your CUDA module system
module load CUDA/11.3.1
echo "python3"    

# Activate conda environment
# NOTE: Change this path according to your conda installation and environment
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator

# Set environment variables
export WANDB_DIR=/tmp
export WANDB_EXECUTABLE=$(which python)
# NOTE: Change this library path according to your conda installation
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/
echo "python4"    
# Run the Python script with relative paths

python "$SCRIPT_DIR/peak_matching_script.py" "$INPUT_JSON" 2>&1

echo "python5"
