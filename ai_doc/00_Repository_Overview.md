# ChemStructLLM Repository Overview

This repository contains a comprehensive suite of tools and models for chemical structure analysis, prediction, and optimization using various machine learning approaches, particularly focused on spectral data interpretation and molecular optimization.

## Repository Structure

The repository is organized into several key components:

1. **LLM_Structure_Elucidator**: Large Language Model-based approach for elucidating chemical structures from spectral data
2. **Smiles-TO-iUpac-Translator**: Translation tool between SMILES notation and IUPAC nomenclature
3. **chemformer_public**: Implementation of the Chemformer model for molecular transformations
4. **chemprop-IR**: Chemical property prediction from IR spectral data
5. **deep-molecular-optimization**: Deep learning framework for molecular optimization
6. **nmr_sgnn_norm**: NMR data analysis using Spatial Graph Neural Networks with normalization
7. **utils_MMT**: Utility functions for Multi-Modal Transformer models

## Key Files

- **JSON Files**: `itos.json`, `stoi.json`, `itos_MF.json`, `stoi_MF.json` - Mapping files for tokenization (index-to-string and string-to-index)
- **Notebooks**: Several Jupyter notebooks for data analysis, testing, and visualization
- **requirements.txt**: Python dependencies for the project

## Integration

The components work together to provide a comprehensive pipeline for chemical structure analysis:
1. Raw spectral data (NMR, IR) is processed through specialized models
2. Molecular representations are generated and optimized
3. Properties are predicted and structures are elucidated
4. Results can be translated between different chemical notation systems

Each folder contains specialized code for different aspects of the chemical structure analysis pipeline, with the utils_MMT folder providing shared functionality across components.
