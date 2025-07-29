#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Set up logging while showing output in console
LOG_FILE="./logs/sgnn_sbatch_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

echo "SBATCH";
echo "Starting SGNN Script at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Running on Windows environment"
    # Windows-specific Python execution
    python "$SCRIPT_DIR/SGNN_script.py" "$@" 2>&1
else
    echo "Running on Linux/HPC environment"
    # HPC-specific setup
    module purge
    echo $(which nvcc)

    # Load Anaconda module and activate environment
    echo "ACTIVATE";
    # NOTE: Change these paths according to your conda installation and environment
    source "$SCRIPT_DIR/../miniconda_SE/bin/activate" "$SCRIPT_DIR/../miniconda_SE/envs/NMR_Structure_Elucidator" 
    # NOTE: Change this according to your CUDA module system
    module load CUDA/11.3.1
    echo $(which python)
    export WANDB_DIR=/tmp
    export WANDB_EXECUTABLE=$(which python)

    echo "nvidia-smi";    
    # nvidia-smi
    # NOTE: Change this library path according to your conda installation
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$SCRIPT_DIR/../miniconda_all/lib/"
    echo "python";
    # Run the Python script with relative paths
    python "$SCRIPT_DIR/SGNN_script.py" "$@" 2>&1
fi

echo "SGNN Script completed at $(date)"