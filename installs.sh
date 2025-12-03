#!/bin/bash

echo "========================================="
echo "Installing packages for NMR_Structure_Elucidator"
echo "========================================="
echo ""
echo "IMPORTANT: Make sure you have activated the environment first:"
echo "    conda activate NMR_Structure_Elucidator"
echo ""

# Check if correct environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "NMR_Structure_Elucidator" ]]; then
    echo "ERROR: NMR_Structure_Elucidator environment is not activated!"
    echo "Please run: conda activate NMR_Structure_Elucidator"
    exit 1
fi

echo "✓ Environment verified: $CONDA_DEFAULT_ENV"
echo ""
echo "========================================="
echo "Installing Jupyter packages..."
echo "========================================="

pip install jupyter-client==7.0.6
pip install jupyter_core==4.11.1
pip install jupyter-server==1.23.4
pip install jupyterlab==3.5.0
pip install jupyterlab-pygments==0.2.2
pip install jupyterlab_server==2.19.0
pip install jupyterlab-widgets==3.0.7
pip install jupyter_core==4.11.1
conda install -y -c conda-forge nbclassic=0.5.1
conda install -y -c conda-forge nbconvert=7.2.9
conda install -y -c conda-forge nbformat=5.7.3
conda install -y -c conda-forge nbclient=0.6.8
conda install -y nb_conda

echo ""
echo "========================================="
echo "Installing PyTorch and Deep Learning packages..."
echo "========================================="

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==0.7.3
pip install six==1.16.0
pip install setuptools==59.5.0
pip install dgl-cu111==0.6.1

echo ""
echo "========================================="
echo "Installing visualization and utilities..."
echo "========================================="

pip install svgwrite
pip install wandb==0.15.4
pip install tqdm

echo ""
echo "========================================="
echo "Installing Flask and web dependencies..."
echo "========================================="

pip install Flask==2.2.5
pip install Flask-SocketIO==5.3.6

echo ""
echo "========================================="
echo "Installing scientific packages..."
echo "========================================="

pip install pandas==1.3.5
pip install matplotlib==3.5.3
pip install plotly==5.18.0

echo ""
echo "========================================="
echo "Installing chemistry packages..."
echo "========================================="

pip install umap-learn==0.5.4
pip install rdkit==2023.3.2
pip install ipython==7.15.0
pip install CairoSVG==2.7.0

pip install MolVS==0.1.1
pip install dgllife==0.3.0
pip install transformers==4.26.1
pip install ipykernel==6.15.0 ipython==7.33.0
pip install tensorboardX==2.6.2.2

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "To activate this environment, run:"
echo "    conda activate NMR_Structure_Elucidator"
echo ""
echo "To verify installation, run:"
echo "    python -c 'import torch; import dgl; import rdkit; print(\"✅ All packages imported successfully!\")'"
echo ""

