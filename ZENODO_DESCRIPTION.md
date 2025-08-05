# Zenodo Dataset Description: ChemStructLLM

## Title
**Enhancing Molecular Structure Elucidation with Reasoning-Capable LLMs - Dataset and Pre-trained Models**

## Creators
- **Priessner, Martin** (Contact person) - ORCID: [Your ORCID ID]
- **Lewis, R.J.**
- **Johansson, M.J.**
- **Goodman, J.M.**
- **Janet, J.P.**
- **Tomberg, A.**

## Description

This Zenodo repository contains the essential data files and pre-trained models required to reproduce the experimental results from our paper "Enhancing Molecular Structure Elucidation with Reasoning-Capable LLMs" (Priessner et al., 2025).

### Contents Overview

This dataset provides three main components necessary for full system reproduction:

#### 1. Pre-trained Models (`models.zip`)
Contains all trained machine learning models used in the ChemStructLLM pipeline:

- **Chemformer Models**: Forward and retrosynthesis prediction models
  - Location: `models/chemformer/`
  - Pre-trained transformer models for molecular transformation prediction
  
- **MMST (Multimodal Spectrum Transformer) Models**: 
  - Location: `models/mmst/base_models/`
  - Fine-tuned models for spectral data analysis
  - Includes HSQC-specific fine-tuned variants
  
- **Mol2Mol Models**: 
  - Location: `models/mol2mol/Alessandro_big/`
  - Molecular transformation and structure generation models

#### 2. Molecular Data and Test Cases (`data_molecular_structure.zip`)
Experimental and reference data for system validation:

- **Molecular Database**: `data/molecular_data/molecular_data.json`
  - Complete spectroscopic data (1H NMR, 13C NMR, HSQC, COSY) for 34 diverse organic molecules
  - Experimental NMR measurements in structured JSON format
  - SMILES representations and molecular identifiers
  
- **Archive Data**: `data/molecular_data/archive/`
  - Historical data versions and backup files
  - Additional molecular datasets used during development

#### 3. Pre-computed Workflow Results (`intermediate_results_selected.zip`)
Example workflow outputs for analysis and validation:

- **Simulation Results**: 
  - `_run_1_sim_finished/` & `_run_1_sim_finished_clean/`: Basic simulation workflow
  - `_run_2_sim_aug_finished/` & `_run_2_sim_aug_finished_clean/`: Augmented simulation data
  - `_run_3_sim+noise_finished/` & `_run_3_sim+noise_finished_clean/`: Noise-augmented simulations
  
- **Experimental Results**:
  - `_run_4_exp_d1_finished/` & `_run_4_exp_d1_finished_clean/`: Experimental dataset 1 results
  - `_run_5_exp_d1_aug_finished/` & `_run_5_exp_d1_aug_finished_clean/`: Augmented experimental results
  - `_run_6_exp_d4_finished/` & `_run_6_exp_d4_finished_clean/`: Experimental dataset 4 results

Each result folder contains:
- Structure elucidation outputs
- LLM reasoning traces
- Performance metrics and evaluation data
- Intermediate analysis files

### System Requirements

- **Operating System**: Linux/macOS (tested on Ubuntu 20.04+, macOS 12+)
- **Python**: 3.7+ (multiple environments required)
- **GPU**: CUDA-compatible GPU recommended for model inference
- **Storage**: ~10GB for complete dataset
- **Memory**: 16GB+ RAM recommended

### Installation and Usage

1. **Download all three zip files** from this Zenodo repository
2. **Extract to correct locations** in your ChemStructLLM installation:
   ```bash
   # Extract models to root directory
   cd ChemStructLLM_/
   unzip models.zip
   
   # Extract data to LLM_Structure_Elucidator directory
   cd LLM_Structure_Elucidator/
   unzip data_molecular_structure.zip
   unzip intermediate_results_selected.zip
   ```
3. **Follow setup instructions** in the main repository README.md

### Research Applications

This dataset enables researchers to:

- **Reproduce Paper Results**: All experimental data and pre-trained models included
- **Benchmark New Methods**: Use our molecular dataset for comparative studies
- **Extend the Framework**: Build upon our pre-trained models and workflows
- **Analyze LLM Performance**: Study reasoning traces and confidence scores across different models

### Technical Specifications

- **Data Format**: JSON for molecular data, PyTorch models for ML components
- **Spectral Data**: Industry-standard NMR formats with chemical shift and coupling information
- **Model Architecture**: Transformer-based models with multimodal capabilities
- **Evaluation Metrics**: Structure recovery accuracy, confidence calibration, reasoning quality

### Additional Components

This upload also includes supplementary work in two specialized areas:

1. **MMT Explainability** (`mmt_explainability`): 
   - Multimodal spectrum transformer fine-tuned specifically on HSQC data
   - Interpretability analysis tools and visualizations
   
2. **MMT Identifier** (`mmt_identifier`): 
   - Latent space investigations using the multimodal spectrum transformer
   - Molecular embedding analysis and clustering studies

## Related Links

- **Full Code Repository**: https://github.com/mpriessner/ChemStructLLM_
- **Publication**: Priessner, M., et al. (2025). Enhancing Molecular Structure Elucidation with Reasoning-Capable LLMs. ChemRxiv
- **Documentation**: Complete setup and usage guide available in repository

## Citation

If you use this dataset or models in your research, please cite:

```bibtex
@article{priessner2025chemstructllm,
  title={Enhancing Molecular Structure Elucidation with Reasoning-Capable LLMs},
  author={Priessner, Martin and Lewis, R.J. and Johansson, M.J. and Goodman, J.M. and Janet, J.P. and Tomberg, Anna},
  journal={ChemRxiv},
  year={2025},
  doi={[DOI when available]}
}
```

## Contact Information

For questions regarding the dataset, models, or implementation:

- **Primary Contact**: martin.priessner@gmail.com
- **Corresponding Author**: anna.tomberg@astrazeneca.com

## License

This dataset is released under [specify license - e.g., Creative Commons Attribution 4.0 International License].

## Acknowledgments

We thank the contributors to the open-source chemical informatics community and the developers of the underlying machine learning frameworks that made this work possible.

---

**Note**: This dataset is designed to work seamlessly with the ChemStructLLM codebase. For optimal results, please use the exact versions and configurations specified in the main repository documentation.
