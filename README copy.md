# MMT_explainability

## Environment Setup

This project requires multiple conda environments for different components. Below are the instructions for setting up each environment.

### Main Environment

```bash
# Create and activate the main environment
conda create -n mmt_explainability python=3.9
conda activate mmt_explainability
pip install -r requirements.txt
```

### STOUT Environment (SMILES to IUPAC Translation)

```bash
# Create and activate the STOUT environment
conda create -n stout_env python=3.9
conda activate stout_env

# Clone and install STOUT
git clone https://github.com/Kohulan/Smiles-TO-iUpac-Translator.git
cd Smiles-TO-iUpac-Translator
pip install .
cd ..
```

### Additional Required Environments

**TODO**: The following environments need to be documented and automated:
- SGNN environment setup
- Other specialized environments

## Future Improvements

- [ ] Create an automated setup script to handle all environment creation and dependency installation
- [ ] Implement environment management through configuration files
- [ ] Add environment validation and testing scripts
- [ ] Document environment-specific requirements and dependencies

## Usage

[TODO: Add usage instructions]

## Contributing

[TODO: Add contribution guidelines]

## License

[TODO: Add license information]