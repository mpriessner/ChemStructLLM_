
#sys.path.append('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer')
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Build the path to the MultiModalTransformer directory relative to the script's location
base_path = os.path.abspath(os.path.join(script_dir, '../../'))

# Add the MultiModalTransformer directory to sys.path
if base_path not in sys.path:
    sys.path.append(base_path)

import numpy as np
import pandas as pd
import random
import operator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles, Descriptors, AllChem, DataStructs

import utils_MMT.helper_functions_pl_v15_4 as hf
import utils_MMT.train_test_functions_pl_v15_4 as ttf
import utils_MMT.sgnn_code_pl_v15_4 as sgnn
from utils_MMT.models_MMT_v15_4 import MultimodalTransformer, TransformerMultiGPU
from utils_MMT.dataloaders_pl_v15_4 import MultimodalData, collate_fn

import wandb
import statistics
from collections import defaultdict
from functools import reduce


SEED = 42
SEED = random.randint(0,1000)
print(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2**32))
    random.seed(torch.initial_seed() % (2**32))
    

def load_data_and_MMT_model(config, stoi, stoi_MF, single=True, mode="val"):
    """Loads the dataset and Multimodal Transformer (MMT) model."""
    # Initialize and load the multi-GPU model
    multi_gpu_model = TransformerMultiGPU(config)
    multi_gpu_model = multi_gpu_model.load_from_checkpoint(config.checkpoint_path, config=config)
    multi_gpu_model.model.to("cuda")

    # Load and prepare the validation dataset
    data = MultimodalData(config, stoi, stoi_MF, mode=mode)
    if single:
        dataloader = DataLoader(data, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    collate_fn=collate_fn,
                                    drop_last=True, 
                                    worker_init_fn=worker_init_fn)
    else:
        dataloader = DataLoader(data, 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn,
                            drop_last=False, 
                            worker_init_fn=worker_init_fn)
    return multi_gpu_model.model, dataloader


def _generate_embeddings(model, src, mask, embedding_function):
    """Helper function to generate embeddings."""
    if src is not None:
        return embedding_function(src, mask)
    return None, None


def run_model(model, data_dict, config):
    # Model preparation and data loading for GPU
    x = data_dict
    src_1H = x['src_1H'].to(config.device) if "1H" in config.training_mode else None
    mask_1H = x['mask_1H'].to(config.device) if "1H" in config.training_mode else None
    src_13C = x['src_13C'].to(config.device) if "13C" in config.training_mode else None
    mask_13C = x['mask_13C'].to(config.device) if "13C" in config.training_mode else None
    src_HSQC = x['src_HSQC'].to(config.device) if "HSQC" in config.training_mode else None
    mask_HSQC = x['mask_HSQC'].to(config.device) if "HSQC" in config.training_mode else None
    src_COSY = x['src_COSY'].to(config.device) if "COSY" in config.training_mode else None
    mask_COSY = x['mask_COSY'].to(config.device) if "COSY" in config.training_mode else None
    src_IR = x['src_IR'].to(config.device) if "IR" in config.training_mode else None
    mask_IR = x['mask_IR'].to(config.device) if "IR" in config.training_mode else None
    src_MF = x['src_MF'].to(config.device) if "MF" in config.training_mode else None
    mask_MF = x['mask_MF'].to(config.device) if "MF" in config.training_mode else None
    src_MS = x['src_MS'].to(config.device) if "MS" in config.training_mode else None
    mask_MS = x['mask_MS'].to(config.device) if "MS" in config.training_mode else None      
    trg_MW = x['trg_MW'].to(config.device) if "MW" in config.training_mode else None    
    trg_enc_SMI = x['trg_enc_SMI']
    src_HSQC_ = x['src_HSQC_'].to(config.device) # bacause otherwise breaks the code
    src_COSY_ = x['src_COSY_'].to(config.device) # bacause otherwise breaks the code


    trg_MW = trg_MW.unsqueeze(1)       
    with torch.no_grad():
        # Create embeddings
        embedding_src_1H, src_padding_mask_1H = model._embed_spectrum_1H(src_1H, mask_1H) if "1H" in config.training_mode else (None, None)
        current_batch_size = embedding_src_1H.shape[1] if "1H" in config.training_mode else False

        embedding_src_13C, src_padding_mask_13C = model._embed_spectrum_13C(src_13C, mask_13C) if "13C" in config.training_mode else (None, None)
        current_batch_size = embedding_src_13C.shape[1] if "13C" in config.training_mode else current_batch_size

        embedding_src_HSQC, src_padding_mask_HSQC = model._embed_spectrum_HSQC(src_HSQC, mask_HSQC) if "HSQC" in config.training_mode else (None, None)
        current_batch_size = embedding_src_HSQC.shape[1] if "HSQC" in config.training_mode  else current_batch_size

        embedding_src_COSY, src_padding_mask_COSY = model._embed_spectrum_COSY(src_COSY, mask_COSY) if "COSY" in config.training_mode else (None, None)
        current_batch_size = embedding_src_COSY.shape[1] if "COSY" in config.training_mode else current_batch_size

        embedding_src_IR, src_padding_mask_IR = model._embed_spectrum_IR(src_IR, mask_IR) if "IR" in config.training_mode else (None, None)       

        embedding_src_MF, src_padding_mask_MF = model._embed_MF(src_MF, mask_MF) if "MF" in config.training_mode else (None, None)

        embedding_src_MS, src_padding_mask_MS = model._embed_MS(src_MS, mask_MS) if "MS" in config.training_mode else (None, None)

        embedding_src_MW, src_padding_mask_MW = model._embed_MW(trg_MW) if "MW" in config.training_mode else (None, None)

        if embedding_src_IR != None:
            embedding_src_IR, src_padding_mask_IR = embedding_src_IR.unsqueeze(0), src_padding_mask_IR.unsqueeze(-1)

        if embedding_src_MW != None:
            embedding_src_MW = embedding_src_MW.unsqueeze(0) 

        feature_dim = 193 if "MS" in config.training_mode else 129
        feature_dim_IR = 130 if "MS" in config.training_mode else 66

        memory = []
        embedding_src = []
        src_padding_mask = []
        if embedding_src_1H is not None:
            embedding_src_1H, src_padding_mask_1H = model._create_embeddings_and_masks_1H(embedding_src_1H, src_padding_mask_1H, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )
            memory_1H = model.encoder_1H(embedding_src_1H, src_key_padding_mask=src_padding_mask_1H)
            memory.append(memory_1H)
            embedding_src.append(embedding_src_1H)
            src_padding_mask.append(src_padding_mask_1H)
        else:
            # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
            memory_1H = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
            embedding_src_1H = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
            src_padding_mask_1H = torch.ones((current_batch_size, feature_dim)).to(config.device)

            memory.append(memory_1H)
            embedding_src.append(embedding_src_1H)
            src_padding_mask.append(src_padding_mask_1H)
            
        if embedding_src_13C is not None:
            embedding_src_13C, src_padding_mask_13C = model._create_embeddings_and_masks_13C(embedding_src_13C, src_padding_mask_13C, 
                                                                        embedding_src_MF, src_padding_mask_MF,
                                                                        embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                        embedding_src_MW, src_padding_mask_MW,
                                                                        )
            memory_13C = model.encoder_13C(embedding_src_13C, src_key_padding_mask=src_padding_mask_13C)
            memory.append(memory_13C)            
            embedding_src.append(embedding_src_13C)
            src_padding_mask.append(src_padding_mask_13C)
        else:
            # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
            memory_13C = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
            embedding_src_13C = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
            src_padding_mask_13C = torch.ones((current_batch_size, feature_dim)).to(config.device)

            memory.append(memory_13C)
            embedding_src.append(embedding_src_13C)
            src_padding_mask.append(src_padding_mask_13C)   
            
        if embedding_src_HSQC is not None:
            embedding_src_HSQC, src_padding_mask_HSQC = model._create_embeddings_and_masks_HSQC(embedding_src_HSQC, src_padding_mask_HSQC, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )    
            memory_HSQC = model.encoder_HSQC(embedding_src_HSQC, src_key_padding_mask=src_padding_mask_HSQC)
            memory.append(memory_HSQC) 
            embedding_src.append(embedding_src_HSQC)
            src_padding_mask.append(src_padding_mask_HSQC)
        else:
            # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
            memory_HSQC = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
            embedding_src_HSQC = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
            src_padding_mask_HSQC = torch.ones((current_batch_size, feature_dim)).to(config.device)

            memory.append(memory_HSQC)
            embedding_src.append(embedding_src_HSQC)
            src_padding_mask.append(src_padding_mask_HSQC) 
            
        if embedding_src_COSY is not None:
            embedding_src_COSY, src_padding_mask_COSY = model._create_embeddings_and_masks_COSY(embedding_src_COSY, src_padding_mask_COSY, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )    
            memory_COSY = model.encoder_COSY(embedding_src_COSY, src_key_padding_mask=src_padding_mask_COSY)
            memory.append(memory_COSY) 
            embedding_src.append(embedding_src_COSY)
            src_padding_mask.append(src_padding_mask_COSY)
        else:
            # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
            memory_COSY = torch.zeros((65, current_batch_size, 128)).to(config.device)
            embedding_src_COSY = torch.zeros((65, current_batch_size, 128)).to(config.device)
            src_padding_mask_COSY = torch.ones((current_batch_size, 65)).to(config.device)

            memory.append(memory_COSY)
            embedding_src.append(embedding_src_COSY)
            src_padding_mask.append(src_padding_mask_COSY) 
            
        if embedding_src_IR is not None:
            embedding_src_IR, src_padding_mask_IR = model._create_embeddings_and_masks_IR(embedding_src_IR, src_padding_mask_IR, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )
            memory_IR = model.encoder_IR(embedding_src_IR, src_key_padding_mask=src_padding_mask_IR)
            memory.append(memory_IR)  
            embedding_src.append(embedding_src_IR)
            src_padding_mask.append(src_padding_mask_IR)                          
        else:
            # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
            memory_IR = torch.zeros((feature_dim_IR, current_batch_size, 128)).to(config.device)
            embedding_src_IR = torch.zeros((feature_dim_IR, current_batch_size, 128)).to(config.device)
            src_padding_mask_IR = torch.full((current_batch_size, feature_dim_IR), False, dtype=torch.bool).to(config.device)

            memory.append(memory_IR)
            embedding_src.append(embedding_src_IR)
            src_padding_mask.append(src_padding_mask_IR)  

        memory = torch.cat(memory, dim=0) 
        #import IPython; IPython.embed();
        embedding_src = torch.cat(embedding_src, dim=0) 
        src_padding_mask = torch.cat(src_padding_mask, dim=1) 
        memory = model.encoder_cross(memory, src_key_padding_mask=src_padding_mask)
        average_memory = torch.mean(memory, dim=0) 

        fingerprint = model.fp1(average_memory)        

        if src_HSQC==None:
            src_HSQC = src_HSQC_
        if src_COSY==None:
            src_COSY = src_COSY_            
        #print(src_padding_mask.shape)
    return memory, src_padding_mask, trg_enc_SMI.to(config.device), fingerprint, src_HSQC, src_COSY
    
'''
def run_model(model, data_dict, config):
    """Prepares and runs the model for a given batch of data."""
    x = data_dict
    current_batch_size = None
    embeddings_and_masks = []

    # List of all possible spectra types
    spectra_types = ["1H", "13C", "HSQC", "COSY", "IR", "MF", "MS", "MW"]

    # Generate embeddings for each spectrum type if it's included in the training mode
    for spectrum in spectra_types:
        src = x.get(f'src_{spectrum}')
        mask = x.get(f'mask_{spectrum}')
        embedding, mask = _generate_embeddings(model, src.to(config.device), mask.to(config.device), 
                                               getattr(model, f'_embed_spectrum_{spectrum}') if spectrum != "MF" else model._embed_MF)
        if embedding is not None:
            current_batch_size = current_batch_size or embedding.shape[1]
            embeddings_and_masks.append((embedding, mask))

    # Handle cases with no embeddings
    feature_dim = 193 if "MS" in config.training_mode else 129
    for _ in range(len(embeddings_and_masks), len(spectra_types)):
        embeddings_and_masks.append((torch.zeros((feature_dim, current_batch_size, 128)).to(config.device), 
                                     torch.ones((current_batch_size, feature_dim)).to(config.device)))

    # Combine and process all embeddings
    all_embeddings = torch.cat([emb for emb, _ in embeddings_and_masks], dim=0)
    all_masks = torch.cat([mask for _, mask in embeddings_and_masks], dim=1)
    average_memory = torch.mean(all_embeddings, dim=0)
    fingerprint = model.fp1(average_memory)

    trg_enc_SMI = x['trg_enc_SMI'].to(config.device)
    return all_embeddings, all_masks, trg_enc_SMI, fingerprint, x.get('src_HSQC')''' 






def predict_prop_correct_max_sequence(model, stoi, memory, src_padding_mask, trg_enc_SMI, gen_num, config):
    """
    Predicts the properties of each token in a sequence generated by a transformer model.

    Parameters:
    model (torch.nn.Module): The transformer model used for prediction.
    stoi (dict): Dictionary mapping tokens to indices.
    memory (torch.Tensor): The memory tensor output from the transformer's encoder.
    src_padding_mask (torch.Tensor): The source padding mask.
    trg_enc_SMI (torch.Tensor): The target encoded Simplified Molecular Input Line Entry System (SMILES).
    gen_num (int): Number of generations for multinomial sampling.
    config (obj): Configuration object containing model and operation settings.

    Returns:
    tuple: Contains tensors for target sequence, correct token probabilities, 
           maximum probability sequence, maximum token probabilities, 
           and multinomial token probabilities.
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Define the initial target tensor with <SOS> tokens
    N = memory.size(1)
    trg_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    trg_tensor_max = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)

    # Token probabilities containers
    corr_token_prob, max_token_prob = [], []

    # Transpose target encoded SMILES and remove <EOS> token
    trg_enc_SMI_T = trg_enc_SMI.transpose(0, 1)
    real_trg = trg_enc_SMI_T[1:, :]
    
    # Iterate over each token in the target sequence
    with torch.no_grad():
        for idx in range(real_trg.shape[0]):
            # Prepare input for the decoder
            gen_seq_length, N = trg_tensor.shape
            gen_positions = torch.arange(gen_seq_length).unsqueeze(1).expand(gen_seq_length, N).to(config.device)
            embedding_gen = model.dropout2(model.embed_trg(trg_tensor) + model.pe_trg(gen_positions))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

            # Generate output from the decoder
            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
            output = model.fc_out(output)
            probabilities = F.softmax(output / config.temperature, dim=2)

            # Process token probabilities
            next_word = torch.argmax(probabilities[-1], dim=1)
            max_prob = probabilities[-1].gather(1, next_word.unsqueeze(-1)).squeeze()
            max_token_prob.append(max_prob)

            # Update target tensor with max probability token
            trg_tensor_max = torch.cat((trg_tensor_max, next_word.unsqueeze(0)), dim=0)

            # # Multinomial sampling###################################### FALSE LOGIC?! Check again!
            # max_prob_multinomial = torch.zeros(1, device=config.device) #
            # for _ in range(gen_num):
            #     sampled_word = torch.multinomial(probabilities[-1], 1)
            #     sel_prob = probabilities[-1].gather(1, sampled_word).squeeze()
            #     max_prob_multinomial = torch.maximum(max_prob_multinomial, sel_prob)

            # multinom_token_prob.append(max_prob_multinomial)

            # Correct token probability
            if idx < real_trg.shape[0]:
                corr_probability = probabilities[-1].gather(1, real_trg[idx].unsqueeze(-1)).squeeze()
                corr_token_prob.append(corr_probability)

            # Update target tensor with actual next token
            next_word = real_trg[idx].unsqueeze(0)
            trg_tensor = torch.cat((trg_tensor, next_word), dim=0)

        # Run just multinomial sampling for x number of times and then calculate the probabilities of the different runs
        multinom_samples = []
        multinom_samples_prob = []
        gen_num = 1 # hardcode this one in because on high numbers it will always converge to the greedy results
        for _ in range(gen_num):
            multinom_token_prob = []
            trg_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
            for idx in range(real_trg.shape[0]):
                # Prepare input for the decoder
                gen_seq_length, N = trg_tensor.shape
                gen_positions = torch.arange(gen_seq_length).unsqueeze(1).expand(gen_seq_length, N).to(config.device)
                embedding_gen = model.dropout2(model.embed_trg(trg_tensor) + model.pe_trg(gen_positions))
                gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

                # Generate output from the decoder
                output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
                output = model.fc_out(output)
                probabilities = F.softmax(output / config.temperature, dim=2)

                multinomial_zero = torch.zeros(1, device=config.device) #
                sampled_word = torch.multinomial(probabilities[-1], 1)
                sel_prob = probabilities[-1].gather(1, sampled_word).squeeze()
                prob_multinomial = torch.maximum(multinomial_zero, sel_prob)
                multinom_token_prob.append(prob_multinomial) #max_prob_multinomial)

                # Update target tensor with actual next token
                next_word = real_trg[idx].unsqueeze(0)
                trg_tensor = torch.cat((trg_tensor, next_word), dim=0)

            multinom_samples.append(sum(multinom_token_prob)/len(multinom_token_prob))
            multinom_samples_prob.append(multinom_token_prob)

    # Select the best multinomial sampling prob
    multinom_samples_tensor = torch.tensor(multinom_samples)
    _, max_index = torch.max(multinom_samples_tensor, 0)
    multinom_token_prob_best = multinom_samples_prob[max_index]

    # Organize and return probabilities
    max_token_prob = torch.stack(max_token_prob)#.transpose(0, 1)
    multinom_token_prob = torch.stack(multinom_token_prob_best)#.transpose(0, 1)
    multinom_token_prob = multinom_token_prob.view(1, -1).squeeze(0)
    # multinom_token_prob = torch.stack(multinom_token_prob)#.transpose(0, 1)
    # multinom_token_prob = multinom_token_prob.squeeze(1)
    corr_token_prob = torch.stack(corr_token_prob)#.transpose(0, 1)

    # Remove <SOS> token from target sequences
    trg_tensor = trg_tensor.transpose(0, 1)[:, 1:]
    trg_tensor_max = trg_tensor_max.transpose(0, 1)[:, 1:]
    return trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob, multinom_token_prob


def predict_prop_correct_max_sequence_2(model, stoi, memory, src_padding_mask, trg_enc_SMI, config):
    """
    Predicts the properties of each token in a sequence generated by a transformer model.

    Parameters:
    model (torch.nn.Module): The transformer model used for prediction.
    stoi (dict): Dictionary mapping tokens to indices.
    memory (torch.Tensor): The memory tensor output from the transformer's encoder.
    src_padding_mask (torch.Tensor): The source padding mask.
    trg_enc_SMI (torch.Tensor): The target encoded Simplified Molecular Input Line Entry System (SMILES).
    gen_num (int): Number of generations for multinomial sampling.
    config (obj): Configuration object containing model and operation settings.

    Returns:
    tuple: Contains tensors for target sequence, correct token probabilities, 
           maximum probability sequence, maximum token probabilities, 
           and multinomial token probabilities.
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Define the initial target tensor with <SOS> tokens
    N = memory.size(1)
    trg_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    trg_tensor_max = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)

    # Token probabilities containers
    corr_token_prob, max_token_prob = [], []

    # Transpose target encoded SMILES and remove <EOS> token
    trg_enc_SMI_T = trg_enc_SMI.transpose(0, 1)
    real_trg = trg_enc_SMI_T[1:, :]
    
    # Iterate over each token in the target sequence
    with torch.no_grad():
        for idx in range(real_trg.shape[0]):
            # Prepare input for the decoder
            gen_seq_length, N = trg_tensor.shape
            gen_positions = torch.arange(gen_seq_length).unsqueeze(1).expand(gen_seq_length, N).to(config.device)
            embedding_gen = model.dropout2(model.embed_trg(trg_tensor) + model.pe_trg(gen_positions))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

            # Generate output from the decoder
            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
            output = model.fc_out(output)
            probabilities = F.softmax(output / config.temperature, dim=2)

            # Process token probabilities
            next_word = torch.argmax(probabilities[-1], dim=1)
            max_prob = probabilities[-1].gather(1, next_word.unsqueeze(-1)).squeeze()
            max_token_prob.append(max_prob)

            # Update target tensor with max probability token
            trg_tensor_max = torch.cat((trg_tensor_max, next_word.unsqueeze(0)), dim=0)

            # Correct token probability
            if idx < real_trg.shape[0]:
                corr_probability = probabilities[-1].gather(1, real_trg[idx].unsqueeze(-1)).squeeze()
                corr_token_prob.append(corr_probability)

            # Update target tensor with actual next token
            next_word = real_trg[idx].unsqueeze(0)
            trg_tensor = torch.cat((trg_tensor, next_word), dim=0)

        


    # Organize and return probabilities
    max_token_prob = torch.stack(max_token_prob)#.transpose(0, 1)
    corr_token_prob = torch.stack(corr_token_prob)#.transpose(0, 1)

    # Remove <SOS> token from target sequences
    trg_tensor = trg_tensor.transpose(0, 1)[:, 1:]
    trg_tensor_max = trg_tensor_max.transpose(0, 1)[:, 1:]
    return trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob

def predict_performance_metric(trg_tensor, corr_token_prob, max_token_prob, multinom_token_prob, stoi):
    """
    Calculates performance metrics for token predictions in sequences.

    Parameters:
    trg_tensor (torch.Tensor): The target tensor containing sequences.
    corr_token_prob (torch.Tensor): Probabilities of correct tokens.
    max_token_prob (torch.Tensor): Probabilities of tokens with maximum likelihood.
    multinom_token_prob (torch.Tensor): Probabilities of tokens chosen via multinomial sampling.

    Returns:
    tuple: Contains two dictionaries - one with aggregated probabilities and another with sample-wise probabilities.
    """

    # Initialize dictionaries for storing results
    prop_dict, sample_dict = {}, {}
    sample_prob_list_corr, sample_prob_list_max, sample_prob_list_multinom = [], [], []
    prob_corr_multi, prob_corr_avg, prob_max_multi, prob_max_avg, prob_multinom_multi, prob_multinom_avg = [], [], [], [], [], []

    # Iterate over each sequence to calculate probabilities
    seq_corr_probs, seq_max_probs, seq_multinom_probs = [], [], []

    for corr_prob, max_prob, multinom_prob, token in zip(corr_token_prob, max_token_prob, multinom_token_prob, trg_tensor[0]):
        # Initialize lists for individual sequence probabilities
        #import IPython; IPython.embed();
        # Iterate over each token in the sequence
        #for idx, (corr_prob, max_prob, multinom_prob, token) in enumerate(zip(corr_probs, max_probs, multinom_probs, tokens)):
        if token == stoi["<EOS>"]:  # End of sequence
            break
        seq_corr_probs.append(corr_prob.item())
        seq_max_probs.append(max_prob.item())
        seq_multinom_probs.append(multinom_prob.item())

    # Calculate and append aggregated probabilities
    prob_corr_multi = reduce(operator.mul, seq_corr_probs, 1)
    prob_corr_avg = statistics.mean(seq_corr_probs) if seq_corr_probs else 0
    prob_max_multi = reduce(operator.mul, seq_max_probs, 1)
    prob_max_avg = statistics.mean(seq_max_probs) if seq_max_probs else 0
    prob_multinom_multi = reduce(operator.mul, seq_multinom_probs, 1)
    prob_multinom_avg = statistics.mean(seq_multinom_probs) if seq_multinom_probs else 0

    # Populate dictionaries with calculated probabilities
    sample_dict = {
        "sample_prob_list_corr": seq_corr_probs,
        "sample_prob_list_max": seq_max_probs,
        "sample_prob_list_multinom": seq_multinom_probs
    }

    prop_dict = {
        "prob_corr_multi": prob_corr_multi,
        "prob_corr_avg": prob_corr_avg,
        "prob_max_multi": prob_max_multi,
        "prob_max_avg": prob_max_avg,
        "prob_multinom_multi": prob_multinom_multi,
        "prob_multinom_avg": prob_multinom_avg
    }

    return prop_dict, sample_dict

'''

def calculate_corr_max_multinom_prob(model, stoi, val_dataloader, gen_num, config):
    """
    Calculates and aggregates the probabilities of correct, maximum likelihood, 
    and multinomial predictions for tokens over a dataset.

    Parameters:
    model (torch.nn.Module): The neural network model used for predictions.
    stoi (dict): Dictionary mapping tokens to indices.
    val_dataloader (DataLoader): Dataloader for the validation set.
    gen_num (int): Number of generations to consider for multinomial predictions.
    config (object): Configuration object containing model and prediction settings.

    Returns:
    tuple: Contains two dictionaries - one with aggregated probabilities and another with sample-wise probabilities.
    """

    # Initialize dictionaries for storing aggregated and sample-wise probabilities
    prob_dict_results = {}
    aggregated_probs = {
        "corr_prob_multi": [],
        "corr_prob_avg": [],
        "max_prob_multi": [],
        "max_prob_avg": [],
        "multinom_prob_multi": [],
        "multinom_prob_avg": []
    }

    # Iterate over each batch in the dataloader
    for idx, data_dict in enumerate(tqdm(val_dataloader)):
        # Run the model and get predictions
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = run_model(model, data_dict, config)
        trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob, multinom_token_prob = \
            predict_prop_correct_max_sequence(model, stoi, memory, src_padding_mask, trg_enc_SMI, gen_num, config)
        
        # Calculate performance metrics for the batch
        prop_dict, sample_dict = predict_performance_metric(trg_tensor, corr_token_prob, max_token_prob, multinom_token_prob, stoi)

        # Aggregate probabilities across all batches
        for key in aggregated_probs:
            aggregated_probs[key].extend(prop_dict[key.replace("prob_", "prob_list_")])

    # Convert lists of aggregated probabilities to summary statistics
    prob_dict_results = {key: sum(vals) for key, vals in aggregated_probs.items()}
    prob_dict_results.update({key + "_avg": statistics.mean(vals) if vals else 0 for key, vals in aggregated_probs.items()})

    return prob_dict_results, sample_dict

'''
def calculate_corr_max_multinom_prob(model, stoi, val_dataloader, gen_num, config):
    
    prob_dict_results = {}
    aggregated_corr_prob_multi = []
    aggregated_corr_prob_avg = []
    aggregated_max_prob_multi = []
    aggregated_max_prob_avg = []
    aggregated_multinom_prob_multi = []
    aggregated_multinom_prob_avg = []
    #for _ in range(2):  # Num_Runs is the number of times you want to run the entire process for randomized smiles

    for idx, data_dict in enumerate(tqdm(val_dataloader)):
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = run_model(model, data_dict, config)
        trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob, multinom_token_prob = predict_prop_correct_max_sequence(model, stoi, memory, src_padding_mask, trg_enc_SMI, gen_num, config)
        prop_dict, sample_dict = predict_performance_metric(trg_tensor, corr_token_prob, max_token_prob, multinom_token_prob, stoi)

        aggregated_corr_prob_multi.append(prop_dict["prob_corr_multi"])
        aggregated_corr_prob_avg.append(prop_dict["prob_corr_avg"])  
        aggregated_max_prob_multi.append(prop_dict["prob_max_multi"])
        aggregated_max_prob_avg.append(prop_dict["prob_max_avg"]) 
        aggregated_multinom_prob_multi.append(prop_dict["prob_multinom_multi"])
        aggregated_multinom_prob_avg.append(prop_dict["prob_multinom_avg"])
        #if idx == 2:
        #    print(idx)
        #    break
    prob_dict_results["aggregated_corr_prob_multi"] = aggregated_corr_prob_multi
    prob_dict_results["aggregated_corr_prob_avg"] = aggregated_corr_prob_avg
    prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
    prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
    prob_dict_results["aggregated_multinom_prob_multi"] = aggregated_multinom_prob_multi
    prob_dict_results["aggregated_multinom_prob_avg"] = aggregated_multinom_prob_avg
    #import IPython; IPython.embed();

    return prob_dict_results, sample_dict


def calculate_corr_max_prob(model, stoi, val_dataloader, gen_num, config):
    
    prob_dict_results = {}
    aggregated_corr_prob_multi = []
    aggregated_corr_prob_avg = []
    aggregated_max_prob_multi = []
    aggregated_max_prob_avg = []
    aggregated_multinom_prob_multi = []
    aggregated_multinom_prob_avg = []
    #for _ in range(2):  # Num_Runs is the number of times you want to run the entire process for randomized smiles

    for idx, data_dict in enumerate(tqdm(val_dataloader)):
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = run_model(model, data_dict, config)
        trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob = predict_prop_correct_max_sequence_2(model, stoi, memory, src_padding_mask, trg_enc_SMI, gen_num, config)
        prop_dict, sample_dict = predict_performance_metric(trg_tensor, corr_token_prob, max_token_prob, multinom_token_prob, stoi)

        aggregated_corr_prob_multi.append(prop_dict["prob_corr_multi"])
        aggregated_corr_prob_avg.append(prop_dict["prob_corr_avg"])  
        aggregated_max_prob_multi.append(prop_dict["prob_max_multi"])
        aggregated_max_prob_avg.append(prop_dict["prob_max_avg"]) 
        aggregated_multinom_prob_multi.append(prop_dict["prob_multinom_multi"])
        aggregated_multinom_prob_avg.append(prop_dict["prob_multinom_avg"])
        #if idx == 2:
        #    print(idx)
        #    break
    prob_dict_results["aggregated_corr_prob_multi"] = aggregated_corr_prob_multi
    prob_dict_results["aggregated_corr_prob_avg"] = aggregated_corr_prob_avg
    prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
    prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
    prob_dict_results["aggregated_multinom_prob_multi"] = aggregated_multinom_prob_multi
    prob_dict_results["aggregated_multinom_prob_avg"] = aggregated_multinom_prob_avg
    #import IPython; IPython.embed();

    return prob_dict_results, sample_dict
         
def calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list):
    """
    Calculates the Tanimoto similarity between generated and target SMILES strings.

    Parameters:
    gen_conv_SMI_list (list): List of generated SMILES strings.
    trg_conv_SMI_list (list): List of target SMILES strings.

    Returns:
    tuple: Mean and standard deviation of Tanimoto similarities, failed pairs, and all Tanimoto scores.
    """
    tanimoto_scores = []
    failed_pairs = []

    for gen_smi, trg_smi in zip(gen_conv_SMI_list, trg_conv_SMI_list):
        try:
            gen_mol = Chem.MolFromSmiles(gen_smi)
            gen_smi_canonical = Chem.MolToSmiles(gen_mol, canonical=True)
            tan_sim = hf.calculate_tanimoto_from_two_smiles(gen_smi_canonical, trg_smi, 512)
            tanimoto_scores.append(tan_sim)
        except:
            tanimoto_scores.append(0)
            failed_pairs.append((gen_smi, trg_smi))

    tanimoto_scores_cleaned = [i for i in tanimoto_scores if i!=0]
    tanimoto_mean = statistics.mean(tanimoto_scores_cleaned) if tanimoto_scores_cleaned else 0
    tanimoto_std_dev = statistics.stdev(tanimoto_scores_cleaned) if len(tanimoto_scores_cleaned) > 1 else 0

    return tanimoto_mean, tanimoto_std_dev, failed_pairs, tanimoto_scores


    
def greedy_sequence(model, stoi, itos, memory, src_padding_mask, config):
    """
    Generates a sequence of tokens using a greedy approach.

    Parameters:
    model (torch.nn.Module): The trained model for sequence generation.
    stoi (dict): Mapping of tokens to indices.
    itos (dict): Mapping of indices to tokens.
    memory (Tensor): Memory tensor from the model.
    src_padding_mask (Tensor): Source padding mask.
    config (object): Configuration containing model parameters.

    Returns:
    tuple: Tensor of generated tokens and their probabilities.
    """
    model.eval()
    N = memory.size(1)
    greedy_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    greedy_token_prob = []

    with torch.no_grad():
        for _ in range(config.max_len):
            gen_seq_length = greedy_tensor.size(0)
            gen_positions = torch.arange(gen_seq_length, device=config.device).unsqueeze(1).expand(gen_seq_length, N)
            embedding_gen = model.dropout2((model.embed_trg(greedy_tensor) + model.pe_trg(gen_positions)))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

           #import IPython; IPython.embed();
            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
           
            output = model.fc_out(output)
            probabilities = F.softmax(output / config.temperature, dim=2)

            ## Capturing the probability of the best next predicted token
            next_word = torch.argmax(probabilities[-1, :, :], dim=1)
            max_prob = probabilities[-1, :, :].gather(1, next_word.unsqueeze(-1)).squeeze()
            greedy_token_prob.append(max_prob)
            next_word = next_word.unsqueeze(0)

            greedy_tensor = torch.cat((greedy_tensor, next_word), dim=0)
            if (next_word == 0).all():
                break
                


    greedy_token_prob = torch.stack(greedy_token_prob) 
    #greedy_token_prob = greedy_token_prob.transpose(0, 1)

    # remove "SOS" token
    greedy_tensor = greedy_tensor[1:,:]
    greedy_token_prob = greedy_token_prob[1:,:]  # previously this was not here now same as in mrtf 

    return greedy_tensor, greedy_token_prob



def evaluate_greedy(model, stoi, itos, val_dataloader, config, randomize=False):
    """
    Evaluates the greedy generation approach over a dataset.

    Parameters:
    model (torch.nn.Module): The trained model for sequence generation.
    stoi (dict), itos (dict): Mappings for tokens and indices.
    val_dataloader (DataLoader): DataLoader for validation dataset.
    config (object): Configuration containing model parameters.
    randomize (bool): Whether to randomize the generated SMILES for comparison.

    Returns:
    tuple: Metrics and results from greedy evaluation, structured in results_dict.
    """    
    gen_conv_SMI_list = []
    trg_conv_SMI_list = []
    src_HSQC_list = []
    src_COSY_list = []
    token_probs_list = []
    results_dict = defaultdict(list)
    run = 0
    # generate all the smiles of trg and greedy gen
    for i, data_dict in enumerate(val_dataloader):
        #import IPython; IPython.embed();
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = run_model(model,
                                                                       data_dict, 
                                                                       config)
        
        greedy_tensor, greedy_token_prob = greedy_sequence(model, stoi, itos, memory, src_padding_mask, config)
        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor.squeeze(1), greedy_token_prob, itos)
        token_probs_list.append(token_probs)
        gen_conv_SMI_list.append(gen_conv_SMI)
        #gen_conv_SMI_list = gen_conv_SMI_list + gen_conv_SMI
        
        trg_enc_SMI = data_dict["trg_enc_SMI"]
        trg_enc_SMI = trg_enc_SMI.transpose(0, 1)
        trg_SMI_input = trg_enc_SMI[1:, :] # Remove <EOS> token from target sequence
        trg_conv_SMI = hf.tensor_to_smiles(trg_SMI_input, itos)
        trg_conv_SMI_list = trg_conv_SMI_list + trg_conv_SMI
        src_HSQC_list.append(src_HSQC)
        src_COSY_list.append(src_COSY)
    # Calculate validity of gen smiles
    validity_term = hf.get_validity_term(gen_conv_SMI_list) 
    # Calculate tanimoto similarity
    if randomize == True:
        random.shuffle(gen_conv_SMI_list)
    tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)

    
    results_dict[run].append({
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'token_probs_list': token_probs_list,
            'validity_term': validity_term,
            'tanimoto_sim': tanimoto_list_all,
            'failed':failed})
        
    return validity_term, tanimoto_mean, tanimoto_std_dev, gen_conv_SMI_list, failed, results_dict, src_HSQC_list, src_COSY_list    
    



def multinomial_sequence(model, stoi, memory, src_padding_mask, config):
    """
    Generates a sequence using multinomial sampling from the model's predictions.

    Parameters:
    model (torch.nn.Module): The trained model.
    stoi (dict): Mapping from string tokens to indices.
    memory (torch.Tensor): The memory tensor from the model's encoder.
    src_padding_mask (torch.Tensor): Source padding mask.
    config (object): Configuration object with model and generation settings.

    Returns:
    tuple: Tuple containing the generated tensor sequence and the token probabilities.
    """
    model.eval()
    batch_size = memory.size(1)
    multinom_tensor = torch.full((1, batch_size), stoi["<SOS>"], dtype=torch.long, device=config.device)
    multinom_token_prob = []

    with torch.no_grad():
        for _ in range(config.max_len):
            gen_seq_length = multinom_tensor.size(0)
            gen_positions = torch.arange(gen_seq_length).unsqueeze(1).expand(gen_seq_length, batch_size).to(config.device)

            embedding_gen = model.dropout2(model.embed_trg(multinom_tensor) + model.pe_trg(gen_positions))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)
            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
            output = model.fc_out(output)

            probabilities = F.softmax(output[-1] / config.temperature, dim=1)
            next_word = torch.multinomial(probabilities, 1).squeeze(1)
            sel_prob = probabilities.gather(1, next_word.unsqueeze(1)).squeeze(1)

            multinom_tensor = torch.cat((multinom_tensor, next_word.unsqueeze(0)), dim=0)
            multinom_token_prob.append(sel_prob)

    multinom_token_prob = torch.stack(multinom_token_prob).transpose(0, 1)
    multinom_tensor = multinom_tensor[1:]  # Removing the initial <SOS> token

    return multinom_tensor, multinom_token_prob


def find_best_amongst_all_runs(results_dict):
    """
    Iterates over each molecule index across all runs to find the highest Tanimoto 
    similarity score and the corresponding SMILES string for each molecule.

    Parameters:
    results_dict (dict): Dictionary containing results from multiple runs.

    Returns:
    dict: Dictionary with the highest Tanimoto similarity scores, selected SMILES strings, 
          and token probabilities.
    """
    highest_scores = []
    selected_smiles = []
    selected_probs = []

    num_molecules = len(results_dict[0][0]["gen_conv_SMI_list"])
    num_runs = len(results_dict)

    for j in range(num_molecules):
        highest_tanimoto_sim = 0
        best_run_index = 0

        for i in range(num_runs):
            smi = results_dict[i][0]["gen_conv_SMI_list"][j]
            tani_sim = results_dict[i][0]["tanimoto_sim"][j]

            if tani_sim > highest_tanimoto_sim:
                highest_tanimoto_sim = tani_sim
                best_run_index = i

        best_smi = results_dict[best_run_index][0]["gen_conv_SMI_list"][j]
        best_probs = results_dict[best_run_index][0]["token_probs_list"][j]

        highest_scores.append(highest_tanimoto_sim)
        selected_smiles.append(best_smi)
        selected_probs.append(best_probs)

    new_results_dict = {
        "highest_tanimoto_sim": highest_scores,
        "gen_conv_SMI_list": selected_smiles,
        "token_probs_list": selected_probs
    }

    return new_results_dict



def evaluate_multinomial(model, stoi, itos, val_dataloader, config, runs, randomize=False):
    """
    Evaluates the performance of multinomial sequence predictions over multiple runs.

    Parameters:
    model (Torch Model): The trained model for prediction.
    stoi (dict): Dictionary mapping string to index.
    itos (dict): Dictionary mapping index to string.
    val_dataloader (DataLoader): Validation data loader.
    config (object): Configuration object with model and evaluation settings.
    runs (int): Number of runs for evaluation.
    randomize (bool): If True, randomizes the order of generated SMILES.

    Returns:
    tuple: A tuple containing the new results dictionary and the src_HSQC list.
    """
    results_dict = defaultdict(list)

    for run in range(runs):
        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list = [], [], [], []

        for data_dict in val_dataloader:
            memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = run_model(model, data_dict, config)
            multinom_tensor, multinom_token_prob = multinomial_sequence(model, stoi, memory, src_padding_mask, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(multinom_tensor, multinom_token_prob, itos)
            
            gen_conv_SMI_list.extend(gen_conv_SMI)
            token_probs_list.extend(token_probs)
            trg_conv_SMI_list.extend(hf.tensor_to_smiles(data_dict["trg_enc_SMI"].transpose(0, 1)[1:], itos))
            src_HSQC_list.append(src_HSQC)

        validity_term = hf.get_validity_term(gen_conv_SMI_list)
        if randomize:
            # random.shuffle(gen_conv_SMI_list)
            combined_lists = list(zip(gen_conv_SMI_list, token_probs_list))
            random.shuffle(combined_lists)
            gen_conv_SMI_list, token_probs_list = zip(*combined_lists)

        tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
        results_dict[run].append({
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'token_probs_list': token_probs_list,
            'validity_term': validity_term,
            'tanimoto_sim': tanimoto_list_all,
            'tanimoto_mean': tanimoto_mean,
            'tanimoto_std_dev': tanimoto_std_dev,
            'failed': failed
        })

    new_results_dict = find_best_amongst_all_runs(results_dict)
    new_results_dict["trg_conv_SMI_list"] = trg_conv_SMI_list

    tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = calculate_tanimoto_similarity(new_results_dict["gen_conv_SMI_list"], trg_conv_SMI_list)
    new_results_dict.update({
        "tanimoto_sim": tanimoto_list_all,
        "tanimoto_mean": tanimoto_mean,
        "tanimoto_std_dev": tanimoto_std_dev,
        "failed": failed
    })

    return new_results_dict, src_HSQC_list


def beam_search_step(model, stoi, beam, memory, src_padding_mask, config, beam_size):
    """
    Performs a single step of the beam search algorithm.

    Parameters:
    model (Torch Model): The trained model for sequence prediction.
    stoi (dict): Dictionary mapping string to index.
    beam (list): List of tuples representing the current beams.
    memory (Tensor): Encoded memory tensor from the model.
    src_padding_mask (Tensor): Source padding mask.
    config (object): Configuration object with model settings.
    beam_size (int): The number of beams to keep.

    Returns:
    list: A list of updated beams for the next iteration.
    """
    new_beam = []
    seen_sequences = set()  # A set to keep track of sequences that have already been extended
    for score, sequence, prob_sequence in beam:
        seq_tuple = tuple(sequence)
        if seq_tuple in seen_sequences:  # Skip sequences that have already been extended
            continue
        seen_sequences.add(seq_tuple)
        
        if sequence[-1] == stoi["<EOS>"]:
            new_beam.append((score, sequence, prob_sequence))
            continue

        trg = torch.tensor(sequence, dtype=torch.long).unsqueeze(1).to(config.device)
        trg_seq_length, N = trg.shape
        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(config.device))

        embedding_trg = model.dropout2((model.embed_trg(trg) + model.pe_trg(trg_positions)))
        trg_mask = model.generate_square_subsequent_mask(trg_seq_length).to(config.device)
        output = model.decoder(embedding_trg, memory, tgt_mask=trg_mask, memory_key_padding_mask=src_padding_mask)
        output = model.fc_out(output)
        
        probabilities = F.softmax(output / config.temperature, dim=2)

        probs = torch.nn.functional.softmax(output[-1, :, :], dim=-1)  # Output shape: [batch_size, vocab_size]
        top_probs, top_indices = torch.topk(probs, beam_size, dim=-1)  # Top probs and indices shape: [batch_size, beam_size]


        for i in range(beam_size):
            new_sequence = sequence + [top_indices[0, i].item()]
            new_prob_sequence = prob_sequence + [top_probs[0, i].item()]  # Store the probability
            new_seq_tuple = tuple(new_sequence)
            if new_seq_tuple in seen_sequences:  # Skip sequences that would result in duplicates
                continue
            seen_sequences.add(new_seq_tuple)
            new_score = score * top_probs[0, i].item()
            new_beam.append((new_score, new_sequence, new_prob_sequence))
            
    new_beam.sort(key=lambda x: x[0], reverse=True)
    return new_beam[:beam_size]


def beam_search(model, stoi, memory, src_padding_mask, config, beam_size):
    """
    Performs beam search for sequence prediction using a trained model.

    Parameters:
    model (Torch Model): The trained model for sequence prediction.
    stoi (dict): Dictionary mapping string to index.
    memory (Tensor): Encoded memory tensor from the model.
    src_padding_mask (Tensor): Source padding mask.
    config (object): Configuration object with model settings.
    beam_size (int): The number of beams to keep.

    Returns:
    list: A list of final beam sequences after the search.
    """
    model.eval()
    N = memory.size(1)
    beams = [[(1, [stoi["<SOS>"]], [])] for _ in range(N)]  # Initialize beams as sequences with only the <SOS> token and score 1 (=100% probability)

    with torch.no_grad():
        for idx in range(0, config.gen_len): # because otherwise the last token gets predicted randomly
            new_beams = []
            new_token_probs = []   
            
            for i in range(N):  # For each item in the batch
                beam = beams[i]
                #prob_beam = token_probs[i]  # Get the token probabilities for this beam
                memory_i = memory[:, i:i+1, :] # Dimensions: ...
                src_padding_mask_i = src_padding_mask[i:i+1] if src_padding_mask is not None else None
                new_beam  = beam_search_step(model, stoi, beam, memory_i, src_padding_mask_i, config, beam_size)
                new_beams.append(new_beam)
                
            beams = new_beams  # Update beams with new sequences and scores
            
    # Extract the sequences and scores from the beams
    beam_sequences = [[item[1] for item in beam] for beam in beams]
    beam_scores = [[item[0] for item in beam] for beam in beams]
    
    return beams


# def evaluate_beam_search(model, stoi, itos, val_dataloader, config,  beam_size, randomize=False):
#     """
#     Evaluates the performance of the beam search method in generating SMILES strings.

#     Parameters:
#     model (Torch Model): Trained model for sequence generation.
#     stoi (dict): String-to-index mapping.
#     itos (dict): Index-to-string mapping.
#     val_dataloader (DataLoader): Validation data loader.
#     config (object): Configuration object with model settings.
#     beam_size (int): Number of beams used in beam search.
#     randomize (bool): Whether to randomize the generated SMILES strings for comparison.

#     Returns:
#     dict: Dictionary containing evaluation results including Tanimoto similarity and validity.
#     """
#     results_dict = defaultdict(list)
#     all_gen_conv_SMI_dict = defaultdict(list)
#     tanimoto_sim_results = defaultdict(list)
#     trg_conv_SMI_list =  []
#     src_HSQC_list = []
#     src_COSY_list = []
#     all_beams = []
#     # Iterate through the validation dataloader
#     for sample, data_dict in enumerate(val_dataloader):
#         memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = run_model(model,
#                                                                        data_dict,
#                                                                        config)
#         # Run beam search
#         beam_sequences = beam_search(model, stoi, memory, src_padding_mask, config, beam_size)

#         beams_tensor = [pad_sequence([torch.tensor(seq[1],dtype=torch.long) for seq in beam]) for beam in beam_sequences]
#         padded_beams = pad_sequence(beams_tensor, batch_first=True, padding_value=0)

#         probs_tensor = [pad_sequence([torch.tensor(seq[2],dtype=torch.float) for seq in beam]) for beam in beam_sequences]
#         padded_probs = pad_sequence(probs_tensor, batch_first=True, padding_value=0)
        
#         # Convert tensor sequences to SMILES
#         batch_size,_, beam_size = padded_probs.shape

#         SMI_prob_list = []
#         for beam in range(beam_size):  # Assuming beam_sequences is a list of lists of lists
#             gen_conv_SMI_gen, token_probs_gen = hf.tensor_to_smiles_and_prob(padded_beams[0][:,beam][1:], padded_probs[0][:,beam], itos)
#             SMI_prob_list.append([gen_conv_SMI_gen, token_probs_gen]) 
#         all_gen_conv_SMI_dict[sample] = SMI_prob_list

#         # Convert target sequences to SMILES
#         trg_enc_SMI = data_dict["trg_enc_SMI"]
#         trg_enc_SMI = trg_enc_SMI.transpose(0, 1)
#         trg_SMI_input = trg_enc_SMI[1:, :]  # Remove <EOS> token from target sequence
#         trg_conv_SMI = hf.tensor_to_smiles(trg_SMI_input, itos)
#         trg_conv_SMI_list =  trg_conv_SMI_list + trg_conv_SMI
#         src_HSQC_list.append(src_HSQC)
#         src_COSY_list.append(src_COSY)

#     # Calculate validity of generated SMILES
#     for beam in range(len(SMI_prob_list)):
#         gen_conv_SMI_list_ = []
#         probs_list_ = []    
#         for sample in range(len(all_gen_conv_SMI_dict)):
#             gen_conv_SMI, probs = all_gen_conv_SMI_dict[sample][beam][0], all_gen_conv_SMI_dict[sample][beam][1]
#             gen_conv_SMI_list_.append(gen_conv_SMI)
#             probs_list_.append(probs)
          
#         validity_term = hf.get_validity_term(gen_conv_SMI_list_)  

#         if randomize == True:
#             combined_lists = list(zip(gen_conv_SMI_list_, probs_list_))
#             random.shuffle(combined_lists)
#             # Unzip the shuffled pairs back into separate lists
#             gen_conv_SMI_list_, probs_list_ = zip(*combined_lists)

#         # Calculate Tanimoto similarity
#         tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = calculate_tanimoto_similarity(gen_conv_SMI_list_, trg_conv_SMI_list)
#         results_dict[beam].append({
#         'gen_conv_SMI_list': gen_conv_SMI_list_, ### put gen_conv_SMI_list here
#         'token_probs_list': probs_list_,
#         'trg_conv_SMI_list': trg_conv_SMI_list,
#         'validity_term': validity_term,
#         'tanimoto_mean': tanimoto_mean,
#         'tanimoto_std_dev': tanimoto_std_dev,
#         'tanimoto_sim': tanimoto_list_all,
#         'failed':failed})

#     new_results_dict = find_best_amongst_all_runs(results_dict)
#     new_results_dict["trg_conv_SMI_list"] = trg_conv_SMI_list  # double?
    
#     tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all= calculate_tanimoto_similarity(new_results_dict["gen_conv_SMI_list"], new_results_dict["trg_conv_SMI_list"])
    
#     new_results_dict["tanimoto_sim"] = tanimoto_list_all
#     new_results_dict["tanimoto_mean"] = tanimoto_mean
#     new_results_dict["tanimoto_std_dev"] = tanimoto_std_dev
#     new_results_dict["failed"] = failed
#     return new_results_dict, src_HSQC_list, src_COSY_list

# Example usage:
# validity_results, tanimoto_sim_results, results_dict = evaluate_beam_search(model, val_dataloader, config, itos, beam_size)

#####################################################################
############ Functions for SGNN similarity comparison ###############
#####################################################################


#Load SGNN mean values
graph_representation = "sparsified"
target = "13C"
#train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H = None, None, None, None
train_y_mean_C, train_y_std_C = sgnn.load_std_mean(target,graph_representation)
target = "1H"
train_y_mean_H, train_y_std_H = sgnn.load_std_mean(target,graph_representation)
sgnn_means_stds = (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)



def calculate_molecular_weights(smiles_list):
    """
    Calculates molecular weights for a list of SMILES strings.

    Parameters:
    smiles_list (list): List of SMILES strings.

    Returns:
    list: Molecular weights corresponding to each SMILES string. None for invalid SMILES.
    """
    molecular_weights = []

    for smiles in tqdm(smiles_list, desc="Calculating Molecular Weights"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.ExactMolWt(mol)
                molecular_weights.append(mw)
            else:
                raise ValueError(f"Invalid SMILES string: {smiles}")
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            molecular_weights.append(None)  # Append None for errors

    return molecular_weights


def prepare_HSQC_data_from_src(src_HSQC_list):
    """
    Processes and scales HSQC spectral data from the source list.

    Parameters:
    src_HSQC_list (list): List containing HSQC spectral data.

    Returns:
    list: A list of processed HSQC tensors.
    """
    processed_HSQC = []

    for src_HSQC in src_HSQC_list:
        for src in src_HSQC:
            # Filter out rows where both elements are not zero
            non_zero_mask = (src != 0).all(dim=1)
            filtered_src = src[non_zero_mask]

            if filtered_src.nelement() != 0:  # Check if tensor is not empty
                scaled_tensors = [filtered_src[:, 0] * 10, filtered_src[:, 1] * 200]
                combined_tensor = torch.stack(scaled_tensors, dim=1)
                processed_HSQC.append(combined_tensor)
            else:
                processed_HSQC.append(torch.tensor([]))  # Append an empty tensor for consistency

    return processed_HSQC


def display_gen_and_trg_molecules(selected_best_SMI_list, trg_conv_SMI_list, show_number):
    """
    Visualizes pairs of generated and target molecules along with their Tanimoto similarity.

    Parameters:
    selected_best_SMI_list (list): List of SMILES strings for the selected best generated molecules.
    trg_conv_SMI_list (list): List of SMILES strings for the target molecules.
    show_number (int): Number of molecule pairs to display.
    """
    # Convert SMILES strings to RDKit Mol objects
    selected_best_mol_list = [Chem.MolFromSmiles(smiles) for smiles in selected_best_SMI_list]
    trg_conv_mol_list = [Chem.MolFromSmiles(smiles) for smiles in trg_conv_SMI_list]

    # Loop through the molecules and create a separate plot for each pair
    for i in range(min(show_number, len(selected_best_mol_list))):
        try:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns per plot

            # Draw molecules
            for mol, ax in zip([selected_best_mol_list[i], trg_conv_mol_list[i]], axs[:2]):
                if mol:
                    img = Draw.MolToImage(mol)
                    ax.imshow(img)
                ax.axis('off')

            # Compute and display Tanimoto similarity if both molecules are valid
            if selected_best_mol_list[i] and trg_conv_mol_list[i]:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(selected_best_mol_list[i], 2, nBits=512)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(trg_conv_mol_list[i], 2, nBits=512)
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                axs[2].text(0.5, 0.5, f'Tanimoto Similarity: {similarity:.2f}', fontsize=12, ha='center', va='center')
            axs[2].axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred for index {i}: {e}")


#####################################################################
##################### Run the full analysis #########################
#####################################################################            
            
# def run_full_analysis(model, stoi, itos, val_dataloader, config): # I could also specify the COSY error
    
#     molecules = {}
#     sample_prob_lists = {}
#     total_results = defaultdict(list)
    
    
#     #### GREEDY Randomize False ####
#     randomize = False
#     #import IPython; IPython.embed();    
#     validity_term, tanimoto_mean, tanimoto_std_dev, gen_conv_SMI_list, failed, results_dict, src_HSQC_list, src_COSY_list = evaluate_greedy(model, stoi, itos, val_dataloader, config,  randomize)

#     # SGNN Evaluation
#     trg_conv_SMI = results_dict[0][0]["trg_conv_SMI_list"]
#     gen_conv_SMI = results_dict[0][0]["gen_conv_SMI_list"]

#     trg_MW = calculate_molecular_weights(trg_conv_SMI)

#     tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = ttf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)


#     if "HSQC" in config.training_mode:
#         tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
#         sgnn_avg_sim_error, sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, sgnn_means_stds, config)
#         total_results["greedy_randomized_False"] = [validity_term, tanimoto_mean, len(failed), sgnn_avg_sim_error]
#         molecules["greedy_randomized_False"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["greedy_randomized_False"]=results_dict[0][0]["token_probs_list"]
#     else:
#         total_results["greedy_randomized_False"] = [validity_term, tanimoto_mean, len(failed), 0]
#         molecules["greedy_randomized_False"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["greedy_randomized_False"]=results_dict[0][0]["token_probs_list"]

    
#     #### GREEDY Randomize True ####
#     randomize = True
#     validity_term, tanimoto_mean, tanimoto_std_dev, gen_conv_SMI_list, failed, results_dict, src_HSQC_list, src_COSY_list = evaluate_greedy(model, stoi, itos, val_dataloader, config,  randomize)

#     # SGNN Evaluation
#     trg_conv_SMI = results_dict[0][0]["trg_conv_SMI_list"]
#     gen_conv_SMI = results_dict[0][0]["gen_conv_SMI_list"]

#     trg_MW = calculate_molecular_weights(trg_conv_SMI)

#     tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = ttf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)
#     if "HSQC" in config.training_mode:
#         tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
#         sgnn_avg_sim_error, sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, sgnn_means_stds, config)
#         total_results["greedy_randomized_True"] = [validity_term, tanimoto_mean, len(failed), sgnn_avg_sim_error]
#         molecules["greedy_randomized_True"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["greedy_randomized_True"]=results_dict[0][0]["token_probs_list"]
        
#     else:
#         total_results["greedy_randomized_True"] = [validity_term, tanimoto_mean, len(failed), 0]
#         molecules["greedy_randomized_True"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["greedy_randomized_True"]=results_dict[0][0]["token_probs_list"]
        
        
#     #### MULTINOMIAL Randomize False ####
#     randomize = False
#     runs = config.multinom_runs

#     #selected_best_SMI_list, trg_conv_SMI_list, highest_scores, optimized_mean_tanimoto_sim, results_dict, src_HSQC_list 
#     new_results_dict, src_HSQC_list = evaluate_multinomial(model, stoi, itos, val_dataloader, config,  runs, randomize)
#     validity_term = hf.get_validity_term(new_results_dict["gen_conv_SMI_list"]) 
#     #failed_num = len([x for x in new_results_dict["tanimoto_sim"] if x == 0])

#     # SGNN Evaluation
#     trg_conv_SMI = new_results_dict["trg_conv_SMI_list"]
#     gen_conv_SMI = new_results_dict["gen_conv_SMI_list"]

#     trg_MW = calculate_molecular_weights(trg_conv_SMI)

#     tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = ttf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)
#     if "HSQC" in config.training_mode:
#         tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
#         sgnn_avg_sim_error, sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, sgnn_means_stds, config)
#         total_results["multinomial_randomized_False"] = [validity_term, tanimoto_mean, len(new_results_dict["failed"]), sgnn_avg_sim_error]
#         molecules["multinomial_randomized_False"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["multinomial_randomized_False"] = new_results_dict["token_probs_list"]
#     else:
#         total_results["multinomial_randomized_False"] = [validity_term, tanimoto_mean, len(new_results_dict["failed"]), 0]
#         molecules["multinomial_randomized_False"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["multinomial_randomized_False"] = new_results_dict["token_probs_list"]
                


     
#     #### MULTINOMIAL Randomize True ####
#     randomize = True
#     runs = config.multinom_runs
#     new_results_dict, src_HSQC_list = evaluate_multinomial(model, stoi, itos, val_dataloader, config,  runs, randomize)
#     validity_term = hf.get_validity_term(new_results_dict["gen_conv_SMI_list"]) 
#     #failed_num = len([x for x in highest_scores if x == 0])

#     # SGNN Evaluation
#     trg_conv_SMI = new_results_dict["trg_conv_SMI_list"]
#     gen_conv_SMI = new_results_dict["gen_conv_SMI_list"]

#     trg_MW = calculate_molecular_weights(trg_conv_SMI)

#     tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = ttf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)
#     if "HSQC" in config.training_mode:
#         tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
#         sgnn_avg_sim_error, sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, sgnn_means_stds, config)
#         total_results["multinomial_randomized_True"] = [validity_term, tanimoto_mean, len(new_results_dict["failed"]), sgnn_avg_sim_error]
#         molecules["multinomial_randomized_True"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["multinomial_randomized_False"] = new_results_dict["token_probs_list"]
#     else:
#         total_results["multinomial_randomized_False"] = [validity_term, tanimoto_mean, len(new_results_dict["failed"]), 0]
#         molecules["multinomial_randomized_True"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["multinomial_randomized_False"] = new_results_dict["token_probs_list"]
           
    
#     ### BEAM  Randomize False ####
#     randomize = False
#     beam_size = config.beam_size
#     #selected_best_SMI_list, trg_conv_SMI_list, highest_scores, optimized_mean_tanimoto_sim, results_dict, src_HSQC_list 
#     new_results_dict, src_HSQC_list = evaluate_beam_search(model, stoi, itos, val_dataloader, config,  beam_size, randomize)
#     validity_term = hf.get_validity_term(new_results_dict["gen_conv_SMI_list"]) 
#     #failed_num = len([x for x in new_results_dict["tanimoto_sim"] if x == 0])

#     # SGNN Evaluation
#     trg_conv_SMI = new_results_dict["trg_conv_SMI_list"]
#     gen_conv_SMI = new_results_dict["gen_conv_SMI_list"]

#     trg_MW = calculate_molecular_weights(trg_conv_SMI)
#     #import IPython; IPython.embed();

#     tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = ttf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)
#     if "HSQC" in config.training_mode:
#         tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
#         sgnn_avg_sim_error, sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, sgnn_means_stds, config)
#         total_results["beam_randomized_False"] = [validity_term, new_results_dict["tanimoto_mean"], len(new_results_dict["failed"]), sgnn_avg_sim_error]
#         molecules["beam_randomized_False"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["beam_randomized_False"]=new_results_dict["token_probs_list"]
#     else:
#         total_results["beam_randomized_False"] = [validity_term, new_results_dict["tanimoto_mean"], len(new_results_dict["failed"]), 0]
#         molecules["beam_randomized_False"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["beam_randomized_False"]=new_results_dict["token_probs_list"]
           
#     ### BEAM  Randomize True ####
#     randomize = True
#     beam_size = config.beam_size
#     new_results_dict, src_HSQC_list = evaluate_beam_search(model, stoi, itos, val_dataloader, config,  beam_size, randomize)
#     validity_term = hf.get_validity_term(new_results_dict["gen_conv_SMI_list"]) 
#     #failed_num = len([x for x in new_results_dict["tanimoto_sim"] if x == 0])

#     # SGNN Evaluation
#     trg_conv_SMI = new_results_dict["trg_conv_SMI_list"]
#     gen_conv_SMI = new_results_dict["gen_conv_SMI_list"]

#     trg_MW = calculate_molecular_weights(trg_conv_SMI)

#     tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = ttf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)
#     if "HSQC" in config.training_mode:
#         tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
#         sgnn_avg_sim_error, sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, sgnn_means_stds, config)
#         total_results["beam_randomized_True"] = [validity_term, new_results_dict["tanimoto_mean"], len(new_results_dict["failed"]), sgnn_avg_sim_error]
#         molecules["beam_randomized_True"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["beam_randomized_True"]=new_results_dict["token_probs_list"]
#     else:
#         total_results["beam_randomized_True"] = [validity_term, new_results_dict["tanimoto_mean"], len(new_results_dict["failed"]), 0]
#         molecules["beam_randomized_True"] = [gen_conv_SMI, trg_conv_SMI]
#         sample_prob_lists["beam_randomized_True"]=new_results_dict["token_probs_list"]
        
#     return total_results, molecules, sample_prob_lists
        


def plot_val_tani_failed_hsqcErr(total_results):
    """
    Plots bar charts for various evaluation metrics from the total_results.

    Parameters:
    total_results (dict): Dictionary containing the results of different evaluation methods.

    """
    data = total_results
    keys_of_interest = ['greedy_randomized_False', 'greedy_randomized_True', 
                        'multinomial_randomized_False', 'multinomial_randomized_True', 
                        'beam_randomized_False', 'beam_randomized_True']
    item_dict = {0: "Validity", 1: "Tanimoto Similarity", 2: "Failed Compounds", 3: "HSQC Error"}

    num_items = len(item_dict)
    num_keys = len(keys_of_interest)
    bar_width = 0.15  # Width of the bars
    index = np.arange(num_keys)  # The x locations for the groups

    fig, axs = plt.subplots(num_items, 1, figsize=(10, num_items * 5), squeeze=False)

    for i in range(num_items):
        values = [data[key][i] if i != 2 else -np.log10(data[key][i]+1) for key in keys_of_interest]  # Log scale for 'Failed Compounds' to improve visibility
        axs[i, 0].bar(index + bar_width * i, values, bar_width, label=item_dict[i])

        # Aesthetics
        axs[i, 0].set_xlabel('Evaluation Methods')
        axs[i, 0].set_ylabel(f'{item_dict[i]}')
        axs[i, 0].set_title(f'Bar Chart for {item_dict[i]}')
        axs[i, 0].set_xticks(index + bar_width / 2)
        axs[i, 0].set_xticklabels(keys_of_interest, rotation=45)
        axs[i, 0].legend()

    plt.tight_layout()
    plt.show()




def plot_corr_max_multinom_stats(total_results):
    """
    Plots bar charts for different statistical measures across correct, max, and multinomial methods.

    Parameters:
    total_results (dict): Dictionary containing the results of different statistical methods.
    """
    data = total_results

    # Relevant keys and labels
    keys_of_interest = ['statistics_multiplication_avg', 'statistics_multiplication_sum', 'statistics_avg_avg']
    labels = ["Correct", "Max", "Multinomial"]

    num_metrics = len(keys_of_interest)
    num_labels = len(labels)
    bar_width = 0.15  # Width of the bars
    index = np.arange(num_labels)  # The x locations for the groups

    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 4), squeeze=False)

    for i, key in enumerate(keys_of_interest):
        values = data[key]
        axs[i, 0].bar(index, values, bar_width, label=key.replace('_', ' ').title())

        # Aesthetics
        axs[i, 0].set_xlabel('Method')
        axs[i, 0].set_ylabel('Value')
        axs[i, 0].set_title(f'Bar Chart for {key.replace("_", " ").title()}')
        axs[i, 0].set_xticks(index)
        axs[i, 0].set_xticklabels(labels, rotation=45)
        axs[i, 0].legend()

    plt.tight_layout()
    plt.show()

    
def run_one_config(config, model, val_dataloader, stoi, itos):
    """
    Executes a series of analysis steps for a given model configuration.

    Parameters:
    config (Config): Configuration object for the model.
    model (Model): Trained model.
    val_dataloader (DataLoader): Validation data loader.
    stoi (dict): String-to-Index mapping.
    itos (dict): Index-to-String mapping.

    Returns:
    tuple: Contains results of analysis, molecule data, sample probabilities, and probability dictionary.
    """
    print(config.training_mode)
    #model, val_dataloader = load_data_and_model(config, stoi, stoi_MF)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total Parameters: {total_params}')
    print(f'Trainable Parameters: {trainable_params}')
    #gen_num = config.multinom_runs
    gen_num = 1
    prob_dict_results, sample_dict = calculate_corr_max_multinom_prob(model, stoi, val_dataloader, gen_num, config)
    try:
        final_prob_max_multi_sum = sum(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_multi_avg = statistics.mean(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_avg_avg = statistics.mean(prob_dict_results["aggregated_max_prob_avg"])
        final_prob_corr_multi_sum = sum(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_multi_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_avg_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_avg"])
        final_prob_multinom_multi_sum = sum(prob_dict_results["aggregated_multinom_prob_multi"])
        final_prob_multinom_multi_avg = statistics.mean(prob_dict_results["aggregated_multinom_prob_multi"])
        final_prob_multinom_avg_avg = statistics.mean(prob_dict_results["aggregated_multinom_prob_avg"])
    except:
        print("failed statistics")
    
    try:     
        
        total_results, molecules, sample_prob_lists = run_full_analysis(model, stoi, itos, val_dataloader, config)

        total_results["statistics_multiplication_avg"] = [final_prob_corr_multi_avg,
                                                final_prob_max_multi_avg,
                                                final_prob_multinom_multi_avg]

        total_results["statistics_multiplication_sum"] = [final_prob_corr_multi_sum,
                                                        final_prob_max_multi_sum,
                                                        final_prob_multinom_multi_sum]

        total_results["statistics_avg_avg"] = [final_prob_corr_avg_avg,
                                                final_prob_max_avg_avg,
                                                final_prob_multinom_avg_avg, ]
    except:
        print("failed run_full_analysis")
        total_results = {}
    try:    
        print("plot_val_tani_failed_hsqcErr")

        plot_val_tani_failed_hsqcErr(total_results)

        print("plot_corr_max_multinom_stats")
        plot_corr_max_multinom_stats(total_results)

        selected_best_SMI_list = molecules["greedy_randomized_False"][0]
        trg_conv_SMI_list = molecules["greedy_randomized_False"][1]
        show_number = 30
        display_gen_and_trg_molecules(selected_best_SMI_list, trg_conv_SMI_list, show_number)
    except:
        print("failed plot_val_tani_failed_hsqcErr")
        
    return total_results, molecules, sample_prob_lists, sample_dict    
    



