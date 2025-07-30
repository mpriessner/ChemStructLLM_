#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
TEMP_DIR="${PROJECT_ROOT}/LLM_Structure_Elucidator/_temp_folder/peak_matching"
LOG_DIR="${TEMP_DIR}/logs"

# Create necessary directories
mkdir -p "$LOG_DIR"

# Enable debug mode
set -x

# Set up logging while showing output in console
LOG_FILE="${LOG_DIR}/peak_matching_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

echo "[$(date)] Starting Peak Matching Script at $(date)"
echo "[$(date)] Running on node: $(hostname)"
echo "[$(date)] Project root: $PROJECT_ROOT"
echo "[$(date)] Temp directory: $TEMP_DIR"

# Get the input JSON file path from temp directory
INPUT_JSON="${TEMP_DIR}/current_run/input_data.json"
if [ ! -f "$INPUT_JSON" ]; then
    echo "[$(date)] ERROR: Input JSON file not found at $INPUT_JSON"
    exit 1
fi

echo "[$(date)] Using input JSON: $INPUT_JSON"

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "[$(date)] Running on Windows environment"
    # Windows-specific Python execution
    echo "[$(date)] Running peak matching script..."
    echo "[$(date)] Python executable: $(which python)"
    echo "[$(date)] Python version: $(python --version)"
    echo "[$(date)] Current directory: $(pwd)"
    echo "[$(date)] PYTHONPATH: $PYTHONPATH"
    echo "[$(date)] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    python "${PROJECT_ROOT}/LLM_Structure_Elucidator/agents/scripts/peak_matching_script.py" "$INPUT_JSON" 2>&1
else
    echo "[$(date)] Running on Linux/HPC environment"
    # HPC-specific setup
    module purge 2>/dev/null || true
    echo $(which nvcc)

    # Load Anaconda module and activate environment
    echo "[$(date)] ACTIVATE"
    source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator
    
    # Try to load CUDA module, fall back to environment variables if it fails
    if ! module load CUDA/11.3.1 2>/dev/null; then
        echo "[$(date)] Module load failed, setting CUDA environment variables directly"
        export PATH=/usr/local/cuda-11.3/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
    fi
    
    echo $(which python)
    export WANDB_DIR=/tmp
    export WANDB_EXECUTABLE=$(which python)

    echo "[$(date)] nvidia-smi"    
    nvidia-smi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/
    
    echo "[$(date)] python"    
    # Change to project root for consistent imports
    cd "$PROJECT_ROOT"
    echo "[$(date)] Changed to project root: $(pwd)"
    
    # Run the Python script
    echo "[$(date)] Running peak matching script..."
    echo "[$(date)] Python executable: $(which python)"
    echo "[$(date)] Python version: $(python --version)"
    echo "[$(date)] Current directory: $(pwd)"
    echo "[$(date)] PYTHONPATH: $PYTHONPATH"
    echo "[$(date)] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    python "${PROJECT_ROOT}/LLM_Structure_Elucidator/agents/scripts/peak_matching_script.py" "$INPUT_JSON" 2>&1
fi

echo "[$(date)] Peak Matching Script completed at $(date)"

# Disable debug mode
set +x
