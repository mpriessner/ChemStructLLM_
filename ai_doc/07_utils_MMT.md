# Utils MMT (Multi-Modal Transformer)

## Overview

The utils_MMT directory contains a comprehensive set of utility functions and modules for Multi-Modal Transformer (MMT) models that process various types of chemical data. This component serves as a shared library that supports multiple other components in the repository, providing common functionality for spectral data processing, model architectures, and chemical data manipulation.

## Key Components

### Core Functionality
- **MT_functions_v15_4.py**: Core Multi-modal Transformer functions
- **models_MMT_v15_4.py**: Multi-modal Transformer model implementations
- **models_CLIP_v15_4.py**: CLIP-inspired models for chemical data
- **dataloaders_pl_v15_4.py**: PyTorch Lightning data loaders for multi-modal data

### Spectroscopic Data Processing
- **nmr_calculation_from_dft_v15_4.py**: NMR calculation utilities from DFT data
- **hsqc_nmr_reconstruction_v15_4.py**: HSQC NMR data reconstruction
- **cosy_nmr_reconstruction_v15_4.py**: COSY NMR data reconstruction
- **ir_simulation_v15_4.py**: IR spectral data simulation

### Model Training and Evaluation
- **train_test_functions_pl_v15_4.py**: Training and testing utilities for PyTorch Lightning
- **helper_functions_pl_v15_4.py**: Helper functions for PyTorch Lightning implementations
- **validate_generate_MMT_v15_4.py**: Validation and generation utilities
- **run_batch_gen_val_MMT_v15_4.py**: Batch generation and validation

### Chemical Data Manipulation
- **smi_augmenter_v15_4.py**: SMILES augmentation utilities
- **similarity_functions_v15_4.py**: Molecular similarity calculation functions
- **sgnn_code_pl_v15_4.py**: Spatial Graph Neural Network implementations
- **molformer_functions_v15_4.py**: Functions for molecular transformer models

### Visualization and Analysis
- **plotting_v15_4.py**: Plotting utilities for chemical data
- **clustering_visualization_v15_4.py**: Clustering and visualization tools
- **mmt_result_test_functions_15_4.py**: Result testing and analysis

### Configuration
- **config_V8.json**: Configuration for general MMT models
- **ir_config_V8.json**: Configuration specific to IR spectral processing

## Integration

The utils_MMT module serves as a central hub of functionality that is used by multiple other components:
- Provides spectral processing for `LLM_Structure_Elucidator`
- Supports model architectures in `nmr_sgnn_norm`
- Offers data handling utilities for `chemprop-IR`
- Implements shared visualization tools used across the repository

## Key Features

- Multi-modal data processing combining spectral and structural information
- Implementation of transformer-based architectures for chemical data
- Specialized functions for different spectroscopic techniques (NMR, IR)
- Integration of graph neural networks with transformer architectures
- Comprehensive evaluation and visualization tools

## Applications

- Supporting structure elucidation from spectral data
- Enabling multi-modal learning across chemical representations
- Providing common utilities for model training and evaluation
- Facilitating data preprocessing and augmentation
- Implementing specialized loss functions and metrics for chemical tasks
