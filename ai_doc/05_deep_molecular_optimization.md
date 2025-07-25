# Deep Molecular Optimization

## Overview

The deep-molecular-optimization component is a comprehensive framework for optimizing molecular structures to achieve desired properties. This module leverages sequence-to-sequence and transformer architectures to learn molecular transformations that improve target properties while maintaining synthetic feasibility.

## Key Components

1. **Model Architectures**: 
   - Sequence-to-sequence models for molecular transformation
   - Transformer models with specialized encoders and decoders for chemical data

2. **Data Processing**:
   - Property change encoding for directed optimization
   - Vocabulary management for chemical language
   - Dataset preparation and augmentation

3. **Training Framework**:
   - Base trainer implementation with common functionality
   - Specialized trainers for different model architectures
   - Optimization strategies for molecular generation

4. **Evaluation and Analysis**:
   - Matched molecular pair analysis
   - Property prediction for generated molecules
   - Visualization tools for molecular transformations

## Directory Structure

- **configuration/**: Configuration management and default settings
- **models/**: 
  - **seq2seq/**: Sequence-to-sequence model implementations
  - **transformer/**: Transformer model components
    - **encode_decode/**: Encoder-decoder architecture components
    - **module/**: Core transformer modules (attention, embeddings, etc.)
- **preprocess/**: Data preparation and preprocessing tools
- **trainer/**: Training implementations for different model architectures
- **utils_MF/**: Utility functions for molecular operations and visualization
- **postprocess/**: Tools for analyzing and visualizing results

## Workflow

1. Molecular data is preprocessed and encoded with property information
2. Models are trained to transform molecules toward desired property profiles
3. Generated molecules are evaluated for property improvements and validity
4. Results are analyzed through MMP analysis and visualization tools

## Integration

This module integrates with other components in the repository:
- Can use `chemformer_public` approaches for certain transformations
- Works with `Smiles-TO-iUpac-Translator` for human-readable outputs
- Shares utility functions with other components through common libraries

## Applications

- Lead optimization in drug discovery
- Material property optimization
- Targeted molecular design
- Exploring chemical space around promising compounds
- Learning from successful molecular optimizations in literature
