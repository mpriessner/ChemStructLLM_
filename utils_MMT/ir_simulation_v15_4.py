import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Build the path to the chemprop-IR directory relative to the script's location
chemprop_ir_path = os.path.abspath(os.path.join(script_dir, '../chemprop-IR'))

# Add the chemprop-IR directory to sys.path
if chemprop_ir_path not in sys.path:
    sys.path.append(chemprop_ir_path)
    
from argparse import Namespace
import csv
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import os
import pandas as pd
from rdkit import Chem

from chemprop.train.predict import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.train.spectral_loss import roundrobin_sid
from chemprop.features import get_available_features_generators
from argparse import Namespace

from chemprop.train import make_predictions
from chemprop.parsing import modify_predict_args

#get_available_features_generators()


def parse_arguments(hyperparameters):
    parsed_args = {key: val[0] for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)


# Function to generate spectral data from smiles (dummy function for demonstration)
def generate_spectral_data_batch(args, smiles_list):
    avg_preds, predictions_df = make_predictions(args, smiles=smiles_list)
    # TODO: Implement the neural network-based spectral data generation
    return avg_preds, predictions_df


def run_IR_simulation(config, args, mode):
    if mode == "target":
        df = pd.read_csv(config.csv_SMI_targets)
    if mode == "1H":
        df = pd.read_csv(config.csv_1H_path_SGNN)

    # Create a directory to save the spectral data files
    folder_path = config.SGNN_gen_folder_path
    # Get the parent folder path
    parent_folder = os.path.dirname(folder_path)
    output_directory = os.path.join(config.SGNN_gen_folder_path, "IR_data_" + str(config.ran_num))
   
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if len(df) < 64:
        batch_size = len(df)
    else:
        batch_size = 64

    # Step 2 and 3: Process in batches of 64
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        smiles_list = batch['SMILES'].tolist()
        sample_ids = batch['sample-id'].tolist()

        # Generate spectral data for the batch
        spectral_data_batch, predictions_df = generate_spectral_data_batch(args, smiles_list)

        # Save spectral data for each sample in the batch
        for j, spectral_data in enumerate(spectral_data_batch):
            #import IPython; IPython.embed();
            output_file_path = os.path.join(output_directory, f"{sample_ids[j]}.csv")
            pd.DataFrame({'spectra': spectral_data}).to_csv(output_file_path, index=False)

    return output_directory