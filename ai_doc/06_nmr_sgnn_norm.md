# NMR SGNN Norm

## Overview

The nmr_sgnn_norm component implements a Spatial Graph Neural Network (SGNN) approach for Nuclear Magnetic Resonance (NMR) data analysis with normalization techniques. This module specializes in predicting and interpreting NMR chemical shifts using graph-based representations of molecules.

## Key Components

1. **Graph Neural Network Models**:
   - **mpnn_baseline.py**: Message Passing Neural Network baseline implementation
   - **mpnn_proposed.py**: Enhanced MPNN model with spatial awareness for NMR prediction

2. **Data Handling**:
   - **dataset.py**: Dataset implementation for NMR data with normalization
   - **util.py**: Utility functions for data processing and transformation

3. **Training Framework**:
   - **train.py**: Training loop implementation with evaluation
   - **run.sh**: Shell script for executing training with standard parameters

4. **Model Definition**:
   - **model.py**: Core model architecture definition

## Workflow

1. Molecular structures are converted to graph representations
2. NMR spectral data is preprocessed and normalized
3. The SGNN model is trained to predict chemical shifts from molecular graphs
4. Results are evaluated against experimental NMR data
5. Predictions can be used for structure verification or elucidation

## Key Features

- Normalization techniques to handle variations in NMR data
- Spatial awareness in graph representations to capture 3D effects on chemical shifts
- Integration of molecular graph topology with spectroscopic data
- Specialized loss functions for NMR shift prediction

## Integration

This module integrates with other components in the repository:
- Provides specialized NMR analysis that complements the `LLM_Structure_Elucidator`
- Can work alongside `chemprop-IR` for multi-spectral analysis
- Leverages common utilities from `utils_MMT` for data handling

## Applications

- Predicting NMR spectra from molecular structures
- Structure verification through NMR shift prediction
- Supporting structure elucidation workflows
- Identifying structural features from NMR patterns
- Quality control in compound synthesis and identification
