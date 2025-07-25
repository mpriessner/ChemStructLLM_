# Standard library imports
import ast
import json
import pickle
import random

import re
import threading
from ast import literal_eval

# Third-party imports
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Custom utility/module imports
from utils_MMT.dataloaders_pl_v15_4 import MultimodalData, collate_fn
from utils_MMT.nmr_calculation_from_dft_v15_4 import *
from utils_MMT.smi_augmenter_v15_4 import SMILESAugmenter
from utils_MMT.models_MMT_v15_4 import MultimodalTransformer, TransformerMultiGPU


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import os
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.nn import functional as F


tqdm.pandas()

# def make_pl(config):
    
#     ### Load dictionaries:
#     # Load from file
#     with open(config.itos_path, 'r') as f:
#         itos = json.load(f)

#     with open(config.stoi_path, 'r') as f:
#         stoi = json.load(f)
        
#     with open(config.itos_MF_path, 'r') as f:
#         itos_MF = json.load(f)

#     with open(config.stoi_MF_path, 'r') as f:
#         stoi_MF = json.load(f)        

#     # Load Model
#     model = MultimodalTransformer(config, src_pad_idx=0)
        
#     # Define a loss function    
#     SMI_loss_fn = torch.nn.CrossEntropyLoss().to(config.device)         
#     #smiles_loss_fn = nn.NLLLoss(reduction="none", ignore_index=0)
#     MF_loss_fn = torch.nn.CrossEntropyLoss().to(config.device)       #not used yet          
#     MW_loss_fn = torch.nn.MSELoss().to(config.device)    
#     FP_loss_fn = torch.nn.BCEWithLogitsLoss().to(config.device)
    

#     #Load SGNN mean values
#     graph_representation = "sparsified"
#     target = "13C"
#     #train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H = None, None, None, None
#     train_y_mean_C, train_y_std_C = load_std_mean(target,graph_representation)
#     target = "1H"
#     train_y_mean_H, train_y_std_H = load_std_mean(target,graph_representation)
#     sgnn_means_stds = (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)
    
#     return model, SMI_loss_fn, MF_loss_fn, MW_loss_fn, FP_loss_fn, itos, stoi, itos_MF, stoi_MF, sgnn_means_stds


def load_dataloaders(config, stoi, stoi_MF):
    
    train_dataset = MultimodalData(config, stoi, stoi_MF, mode="train")    
    test_dataset = MultimodalData(config, stoi, stoi_MF, mode="test")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=config.num_workers, drop_last=False)
    return train_dataloader, test_dataloader



def run_training_MMT(train_dataloader, test_dataloader, config):
    if not os.path.exists(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    # convert the Namespace to dictionary
    config_dict = vars(config)
    # assuming the model_save_dir is a list with one element
    config_save_path = config.model_save_dir
    # save the dictionary as a JSON file
    random_num = random.randint(0,10000)
    with open(f"{config_save_path}/config_{str(random_num)}.json", 'w') as f:
        json.dump(config_dict, f, indent=4) 

    filepath= config.model_save_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        mode='min',
        filepath=os.path.join(filepath,'model-{epoch:02d}-{loss:.4f}'),
        save_top_k=-1,  # Keeps all checkpoints.
    )

    # If i use that the saving doesn't work well
    profiler = SimpleProfiler()

    wandb_logger = WandbLogger(project=config.project, 
                               log_model='all') # log all new checkpoints during training
    transformer_multi_gpu_model = TransformerMultiGPU(config)

    if config.load_model:
        checkpoint_path = config.checkpoint_path
        transformer_multi_gpu_model = transformer_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)
    if config.use_real_data:
        # Freeze all parameters in the model
        for param in transformer_multi_gpu_model.parameters():
            param.requires_grad = False
        # Unfreeze parameters in self.real_data_linear
        # Check if the real_data_linear layer exists in the model's 'model' attribute
        if hasattr(transformer_multi_gpu_model.model, 'real_data_linear'):
            print("has layer")
            for param in transformer_multi_gpu_model.model.real_data_linear.parameters():
                param.requires_grad = True

    config_dict = vars(config)
    wandb_logger.log_hyperparams(config_dict)
    
    try:
        trainer = pl.Trainer(profiler=profiler,
                             gpus=config.gpu_num, 
                             progress_bar_refresh_rate=10, 
                             accelerator='ddp', 
                             logger=wandb_logger,
                             checkpoint_callback=checkpoint_callback,
                             #callbacks=[checkpoint_callback, batch_checkpoint_callback],  # Add the batch_checkpoint_callback here
                             max_epochs=config.num_epochs,
                             #fast_dev_run=True,
                             #early_stop_callback=early_stopping,
                             #limit_train_batches=1,
                             #limit_val_batches=1
                             )

        trainer.fit(transformer_multi_gpu_model, train_dataloader, test_dataloader)
    except Exception as e:
        print(f"Error occurred: {e}")
        backup_ckpt_path = config.model_save_dir+"/last_backup_checkpoint.ckpt"
        trainer.save_checkpoint(backup_ckpt_path)
        print("Model saved.")


def run_MMT(config, stoi, stoi_MF):
    train_dataloader, test_dataloader = load_dataloaders(config, stoi, stoi_MF)
    run_training_MMT(train_dataloader, test_dataloader, config)
    