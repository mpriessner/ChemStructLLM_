#!/bin/bash
# Shell script to run STOUT processor in dedicated environment

# Get absolute paths
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INPUT_FILE="$1"
OUTPUT_FILE="$2"
MODE="$3"  # forward or reverse

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Running on Windows environment"
    # Windows-specific conda activation
    eval "$(conda shell.bash hook)"
    conda activate stout_env
else
    echo "Running on Linux/HPC environment"
    # HPC-specific setup
    module purge
    module load CUDA/11.3.1
    export TF_ENABLE_ONEDNN_OPTS=0
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export CUDA_VISIBLE_DEVICES=""  # Or specific GPU number you want to use
    # Load Anaconda module and activate environment
    echo "ACTIVATE"
    source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/stout_env
    echo $(which python)
fi

# Run the STOUT script
python "$SCRIPT_DIR/stout_script.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --mode "$MODE"