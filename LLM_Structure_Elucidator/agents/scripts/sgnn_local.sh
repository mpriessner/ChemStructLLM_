#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directories (relative to script location)
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels to LLM_Structure_Elucidator root
CONFIG_BASE_DIR="$(cd "$BASE_DIR/.." && pwd)"  # Go up one more level to ChemStructLLM_ root

# Create logs directory if it doesn't exist
mkdir -p "$BASE_DIR/logs"

# Set up logging while showing output in console
LOG_FILE="$BASE_DIR/logs/sgnn_local_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

echo "SGNN Local Script";
echo "Starting SGNN Script at $(date)"
echo "Running on node: $(hostname)"

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Running on Windows environment"
    # Change to CONFIG_BASE_DIR so Python can find utils_MMT module
    cd "$CONFIG_BASE_DIR"
    # Windows-specific Python execution
    python "$SCRIPT_DIR/sgnn_script.py" "$@" 2>&1
else
    echo "Running on Linux/HPC environment"
    # HPC-specific setup
    module purge
    echo $(which nvcc)

    # Load Anaconda module and activate environment
    echo "ACTIVATE";
    # NOTE: Change these paths according to your conda installation and environment
    source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator  # Change to your conda environment path
    # NOTE: Change this according to your CUDA module system
    module load CUDA/11.3.1  # Modify this according to your CUDA module system
    echo $(which python)
    export WANDB_DIR=/tmp
    export WANDB_EXECUTABLE=$(which python)

    echo "nvidia-smi";    
    # nvidia-smi
    # NOTE: Change this library path according to your conda installation
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_SE/lib/  # Change to your conda lib path
    echo "python";
    
    # Change to CONFIG_BASE_DIR so Python can find utils_MMT module
    cd "$CONFIG_BASE_DIR"
    # Run the Python script with correct filename
    python "$SCRIPT_DIR/sgnn_script.py" "$@" 2>&1
fi

echo "SGNN Script completed at $(date)"