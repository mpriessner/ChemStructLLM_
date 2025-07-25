#!/bin/bash
#SBATCH --job-name=mmst_pred
#SBATCH --output=mmst_%j.out
#SBATCH --error=mmst_%j.err
#SBATCH --time=0:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=short-gpu

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Set up logging while showing output in console
LOG_FILE="./logs/mmst_sbatch_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

echo "SBATCH"
echo "Starting MMST Script at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Running on Windows environment"
    # Windows-specific Python execution
    python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/MMST_script.py "$@" 2>&1
else
    echo "Running on Linux/HPC environment"
    # HPC-specific setup
    module purge
    echo $(which nvcc)

    # Load Anaconda module and activate environment
    echo "ACTIVATE"
    source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator
    module load CUDA/11.3.1
    echo $(which python)
    export WANDB_DIR=/tmp 
    export WANDB_EXECUTABLE=$(which python)

    echo "nvidia-smi"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/
    echo "python"
    # Run the Python script
    python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/mmst_script.py "$@" 2>&1
fi

echo "MMST Script completed at $(date)"
