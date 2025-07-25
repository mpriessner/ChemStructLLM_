# ChemStructLLM: Enhancing Molecular Structure Elucidation with Reasoning-Capable LLMs 

![ChemStructLLM Graphical Abstract](GA_2.png)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15317373.svg)](https://zenodo.org/records/15317373)

This repository contains the implementation of our paper: "Enhancing Molecular Structure Elucidation with Reasoning-Capable LLMs" by Martin Priessner, Richard J. Lewis, Magnus J. Johansson, Jonathan M. Goodman, Jon Paul Janet, and Anna Tomberg.

## Overview

We introduce a novel workflow integrating reasoning-capable language models with specialized chemical analysis tools to enhance molecular structure determination using nuclear magnetic resonance spectroscopy. Our framework combines:

1. **Diverse Candidate Generation**: Using Chemformer, Mol2Mol, and MultiModalSpectralTransformer (MMST) approaches
2. **Quantitative Analysis**: HSQC peak matching and spectral prediction
3. **LLM-Driven Reasoning**: Advanced interpretation of spectral evidence with chemical context

This integrated approach significantly improves structure elucidation accuracy, particularly for noisy or ambiguous spectral data.

## Repository Structure

- **MultiModalSpectralTransformer_HSQC/**: Core implementation of the HSQC spectral analysis and matching
- **MMT_identifier/**: Implementation of the MultiModalSpectralTransformer for structure identification
- **scripts/**: Utility scripts for data processing and analysis
- **examples/**: Example notebooks demonstrating usage of the framework
- **prompts/**: LLM prompt templates used for spectral analysis and reasoning

## Usage

For detailed usage instructions, please refer to the Jupyter notebooks provided in the examples directory. These notebooks contain step-by-step experimental procedures and demonstrations of the framework's capabilities.

## Datasets and Code

The full dataset and code used in our experiments is available on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15317373.svg)](https://zenodo.org/records/15317373)

We recommend downloading the complete repository from Zenodo to ensure all dependencies and files are properly organized.

This includes:
- Experimental NMR data for 34 diverse organic molecules
- Regioisomeric analogs for testing structure recovery
- Simulated spectral data with controlled noise
- All necessary code and model files

## Citation

If you use this code or our methods in your research, please cite our paper:

```bibtex
@article{priessner2025enhancing,
  title={Enhancing Molecular Structure Elucidation with Reasoning-Capable LLMs},
  author={Priessner, Martin and Lewis, Richard J. and Johansson, Magnus J. and Goodman, Jonathan M. and Janet, Jon Paul and Tomberg, Anna},
  journal={},
  volume={},
  pages={},
  year={2025},
  publisher={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We gratefully acknowledge AstraZeneca for their support and funding of the Postdoctoral position, instrumental in the success of this research.
