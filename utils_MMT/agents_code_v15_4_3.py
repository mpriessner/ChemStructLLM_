"""
Enhanced version of agents_code with improved config handling.
This version adds support for both dictionary and object-style configurations.
"""

import os
import io
import re
import ast
import math
import json
import pickle
import random
import logging
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import permutations
from typing import Dict, Any, List, Optional, Tuple, Set
import requests
from typing import Dict, Any
import pandas as pd
import anthropic
from rdkit import Chem
from types import SimpleNamespace

# Import all functions from previous version
from .agents_code_v15_4_2 import *

def generate_shifts_batch(config, smiles_list):
    """Generate NMR shifts for a batch of SMILES strings.
    
    Args:
        config: Configuration object or dictionary containing required parameters
        smiles_list: List of SMILES strings to process
        
    Returns:
        Tuple containing (list_nmr_data, list_nmr_data_all, list_sdf_paths, list_smi)
    """
    output_directory = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/trash_folder"
    csv_file_path = ed.save_smiles_as_csv(smiles_list, output_directory)
    
    # Handle both dict and object configs
    log_file = config.log_file if hasattr(config, 'log_file') else config.get('log_file')
    if log_file:
        custom_log(log_file, f"CSV file saved at: {csv_file_path}")

    # Create a SimpleNamespace config if dict was provided
    if isinstance(config, dict):
        config_obj = SimpleNamespace(**config)
        config_obj.SGNN_csv_gen_smi = csv_file_path
        df_gen_data = dg.run_sgnn(config_obj)
    else:
        config.SGNN_csv_gen_smi = csv_file_path
        df_gen_data = dg.run_sgnn(config)
    
    list_nmr_data = []
    list_nmr_data_all = []
    list_sdf_paths = []
    list_smi = []
    
    for _, row in df_gen_data.iterrows():
        try:
            sdf_path = row['sdf_path']
            list_sdf_paths.append(sdf_path)

            # Generate NMR data
            df_hsqc, df_hsqc_all, _ = ed.run_HSQC_generation(sdf_path)
            df_cosy, _ = ed.run_COSY_generation(sdf_path)
            df_13c, df_13c_all = ed.run_13C_generation(sdf_path)
            df_1h = ed.run_1H_generation(sdf_path)
    
            nmr_data = {
                'HSQC': df_hsqc.to_dict(),
                'COSY': df_cosy.to_dict(),
                '13C': df_13c.to_dict(),
                '1H': df_1h.to_dict()
            }
        
            nmr_data_all = {
                'HSQC': df_hsqc_all.to_dict(),
                'COSY': df_cosy.to_dict(),
                '13C': df_13c_all.to_dict(),
                '1H': df_1h.to_dict()
            }
            
            mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
            smiles = Chem.MolToSmiles(mol)

            list_nmr_data.append(nmr_data)
            list_nmr_data_all.append(nmr_data_all)
            list_smi.append(smiles)            
        except Exception as e:
            print(f"Error processing row: {row}")
            print(f"Error details: {str(e)}")
            continue
    
    return list_nmr_data, list_nmr_data_all, list_sdf_paths, list_smi
