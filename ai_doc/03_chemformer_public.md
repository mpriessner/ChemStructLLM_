# Chemformer Public

## Overview

The chemformer_public directory contains an implementation of the Chemformer model, a transformer-based architecture specifically designed for molecular transformations and chemical reaction prediction. This component leverages advanced natural language processing techniques adapted for chemical structures.

## Key Components

1. **Model Architecture**: Transformer-based neural network adapted for molecular data
2. **Tokenization**: Specialized tokenizers for SMILES and other chemical representations
3. **Training Framework**: Tools for training models on chemical reaction datasets
4. **Inference Tools**: Methods for predicting products of chemical reactions
5. **Evaluation Metrics**: Specialized metrics for assessing chemical prediction quality

## Directory Structure

- **molbart/**: Core implementation of the molecular BART architecture
  - **data/**: Data processing utilities for chemical datasets
  - **models/**: Model architecture definitions
  - **tokenizers/**: Chemical tokenization implementations
  - **utils/**: Utility functions for training and evaluation
- **notebooks/**: Jupyter notebooks for analysis and demonstration
- **service/**: API implementations for model serving
- **tests/**: Test suite for model validation

## Workflow

1. Chemical structures are tokenized into suitable representations
2. The transformer model processes these representations
3. Predictions are generated for target properties or transformations
4. Results are detokenized back into chemical notations

## Integration

This module integrates with other components in the repository:
- Can be used alongside `deep-molecular-optimization` for reaction-based optimization
- Provides molecular transformation capabilities that complement the structural elucidation tools
- Can leverage the tokenization mappings in the root directory (itos.json, stoi.json)

## Applications

- Predicting products of chemical reactions
- Retrosynthesis planning
- Molecular optimization through transformation
- Chemical property prediction
- Reaction condition optimization
