# Standard library imports
import argparse
import json
import os
import random
import sys
import time
from argparse import Namespace
from collections import defaultdict

# Third-party imports
## Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm

## Machine learning and data visualization
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

## RDKit for cheminformatics
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add chemprop to path
chemprop_ir_path = os.path.join(project_root, 'chemprop_IR')
if chemprop_ir_path not in sys.path:
    sys.path.append(chemprop_ir_path)

# Import utility functions
import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.execution_function_v15_4 as ex
import utils_MMT.mmt_result_test_functions_15_4 as mrtf
import utils_MMT.helper_functions_pl_v15_4 as hf

from chemprop.train import make_predictions
from chemprop.parsing import modify_predict_args

# Helper functions
def load_json_dics():
    """Load JSON dictionaries for model vocabulary and mappings."""
    script_dir = os.path.dirname(__file__)
    mmt_dir = os.path.abspath(os.path.join(script_dir, '../../..'))  # Go up to MMT_explainability
    
    # Print paths for debugging
    print(f"Looking for JSON files in: {mmt_dir}")
    
    itos_path = os.path.join(mmt_dir, 'itos.json')
    stoi_path = os.path.join(mmt_dir, 'stoi.json')
    stoi_MF_path = os.path.join(mmt_dir, 'stoi_MF.json')
    itos_MF_path = os.path.join(mmt_dir, 'itos_MF.json')
    
    # Check if files exist
    for path in [itos_path, stoi_path, stoi_MF_path, itos_MF_path]:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
    
    with open(itos_path, 'r') as f:
        itos = json.load(f)
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    with open(stoi_MF_path, 'r') as f:
        stoi_MF = json.load(f)
    with open(itos_MF_path, 'r') as f:
        itos_MF = json.load(f)
    
    return itos, stoi, stoi_MF, itos_MF

def parse_arguments(hyperparameters):
    """Parse hyperparameters into a Namespace object."""
    # If already a Namespace, convert to dict
    if hasattr(hyperparameters, '__dict__'):
        hyperparameters = vars(hyperparameters)
    
    # Process the dictionary
    parsed_args = {
        key: val[0] if isinstance(val, (list, tuple)) else val 
        for key, val in hyperparameters.items()
    }
    return Namespace(**parsed_args)

def load_config(path):
    """Load configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_updated_config(config, path):
    """Save updated configuration to a JSON file."""
    config_dict = vars(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_configs(config_dir=None):
    """Load both IR and main configurations."""
    if config_dir is None:
        script_dir = os.path.dirname(__file__)
        mmt_dir = os.path.abspath(os.path.join(script_dir, '../../..'))  # Go up to MMT_explainability
        config_dir = os.path.join(mmt_dir, 'utils_MMT')
    
    # Print paths for debugging
    print(f"Looking for config files in: {config_dir}")
    
    # Both config files are in config directory
    ir_config_path = os.path.join(config_dir, 'ir_config_V8.json')
    config_path = os.path.join(config_dir, 'config_V8.json')
    
    # Check if files exist
    if not os.path.exists(ir_config_path):
        print(f"Warning: IR config not found at: {ir_config_path}")
        raise FileNotFoundError(f"IR config file not found at {ir_config_path}")
        
    if not os.path.exists(config_path):
        print(f"Warning: Main config not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    print(f"Loading IR config from: {ir_config_path}")
    print(f"Loading main config from: {config_path}")
    
    IR_config_dict = load_config(ir_config_path)
    if IR_config_dict is None:
        raise FileNotFoundError(f"Failed to load IR config from {ir_config_path}")
    
    config_dict = load_config(config_path)
    if config_dict is None:
        raise FileNotFoundError(f"Failed to load main config from {config_path}")
    
    # Parse configs
    IR_config = parse_arguments(IR_config_dict)
    modify_predict_args(IR_config)
    config = parse_arguments(config_dict)
    
    return IR_config, config
