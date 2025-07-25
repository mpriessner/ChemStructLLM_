#!/bin/bash

# Activate the specific conda environment
source /projects/cc/knlr326/miniconda_all/bin/activate /projects/cc/knlr326/miniconda_all/envs/chemformer

# Run the Python script
python -m molbart.predict

# Deactivate the environment
conda deactivate