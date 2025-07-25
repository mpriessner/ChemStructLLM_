#!/bin/bash


mkdir -p ./logs

# Start logging
exec 1>./logs/sgnn_sbatch_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "SBATCH";
echo "Starting SGNN Script at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Running on Windows environment"
    # Windows-specific Python execution
    python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/SGNN_script.py "$@" 2>&1
else
    echo "Running on Linux/HPC environment"
    # HPC-specific setup
    module purge
    echo $(which nvcc)

    # Load Anaconda module and activate environment
    echo "ACTIVATE";
    source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate  /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator 
    module load CUDA/11.3.1
    echo $(which python)
    export WANDB_DIR=/tmp
    export WANDB_EXECUTABLE=$(which python)

    echo "nvidia-smi";    
    nvidia-smi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/
    echo "python";
    # Run the Python script
    python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/SGNN_script.py "$@" 2>&1
fi

echo "SGNN Script completed at $(date)"