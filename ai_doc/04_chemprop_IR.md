# Chemprop-IR

## Overview

Chemprop-IR is a specialized adaptation of the Chemprop framework focused on predicting chemical properties from infrared (IR) spectroscopy data. This component extends graph neural network approaches to handle spectral data inputs, enabling direct property prediction from spectroscopic measurements.

## Key Components

1. **Data Processing**: Tools for handling and preprocessing IR spectral data
2. **Graph Neural Networks**: Modified message passing neural networks for molecular representation
3. **Spectral Loss Functions**: Specialized loss functions for spectral data prediction
4. **Feature Generation**: Methods to extract meaningful features from IR spectra
5. **Evaluation Framework**: Tools for assessing model performance on spectral prediction tasks

## Directory Structure

- **chemprop/**: Core implementation of the chemical property prediction framework
  - **data/**: Data handling utilities for chemical and spectral datasets
  - **features/**: Feature extraction and generation tools
  - **models/**: Model architecture definitions
  - **train/**: Training and evaluation utilities
- **scripts/**: Utility scripts for data processing and analysis
  - **SIS_spectra_similarity.py**: Spectral similarity calculation
  - **aist_spectra_image_processing.py**: Processing spectral images
  - **spectra-related scripts**: Various tools for spectral data manipulation

## Workflow

1. IR spectral data is preprocessed and normalized
2. Molecular graphs are constructed from structural information
3. The model combines spectral features with molecular representations
4. Properties are predicted based on the combined representation
5. Results are evaluated against ground truth data

## Integration

This module integrates with other components in the repository:
- Can use structures from `LLM_Structure_Elucidator` for property prediction
- Complements `nmr_sgnn_norm` by providing IR-specific analysis
- Shares conceptual approaches with other property prediction components

## Applications

- Predicting chemical properties directly from IR spectra
- Quality control in chemical synthesis
- Material property screening
- Structure verification through property prediction
- High-throughput spectral analysis
