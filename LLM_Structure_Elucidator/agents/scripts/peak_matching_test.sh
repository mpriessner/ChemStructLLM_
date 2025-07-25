#!/bin/bash

# Hard-coded paths
PROJECT_ROOT="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability"
INPUT_JSON="${PROJECT_ROOT}/LLM_Structure_Elucidator/_temp_folder/peak_matching/current_run/input_data.json"

# HPC-specific setup - following SGNN pattern exactly
echo "python1"    

module purge
echo "python2"    
module load CUDA/11.3.1
echo "python3"    

# Activate conda environment
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator

# Set environment variables
export WANDB_DIR=/tmp
export WANDB_EXECUTABLE=$(which python)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/
echo "python4"    
# Run the Python script with hardcoded paths


python "${PROJECT_ROOT}/LLM_Structure_Elucidator/agents/scripts/peak_matching_script.py" "$INPUT_JSON" 2>&1

echo "python5"    

