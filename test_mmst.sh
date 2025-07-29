#!/bin/bash

# Test script for MMST
cd /Users/mpriessner/windsurf_repos/ChemStructLLM_

# Create test output directory
mkdir -p test_mmst_output

# Run the MMST script with minimal parameters
./LLM_Structure_Elucidator/agents/scripts/mmst_local.sh \
    --input_csv=test_mmst_input.csv \
    --run_dir=test_mmst_output \
    --config_dir=utils_MMT
