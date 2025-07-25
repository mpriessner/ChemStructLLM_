# Standard library imports
import glob
import json
import os
import random
from collections import defaultdict
from functools import reduce
from argparse import Namespace

# Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm
import statistics
import operator

# Visualization libraries
import matplotlib.pyplot as plt

# PyTorch for deep learning
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

# RDKit for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, MolFromSmiles, MolToSmiles

# Weights & Biases for experiment tracking
import wandb

# Local utilities/modules
import utils_MMT.sgnn_code_pl_v15_4 as sc
import utils_MMT.train_test_functions_pl_v15_4 as ttf
import utils_MMT.helper_functions_pl_v15_4 as hf
import utils_MMT.validate_generate_MMT_v15_4 as vgmmt
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
from utils_MMT.dataloaders_pl_v15_4 import MultimodalData, collate_fn
from utils_MMT.models_MMT_v15_4 import MultimodalTransformer, TransformerMultiGPU
from utils_MMT.models_CLIP_v15_4 import CLIPMultiGPU  # CHANGE WHEN I HAVE A CLIP V8 model trained
# from utils_MMT.models_BLIP_v15_4 import BLIPMultiGPU  # CHANGE WHEN I HAVE A CLIP V8 model trained




def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2**32))
    random.seed(torch.initial_seed() % (2**32))

def prepare_HSQC_data_from_src_2(src_HSQC_list):
    """
    Processes and scales HSQC spectral data from the source list.
    """
    processed_HSQC = []

    for src in src_HSQC_list:
        #for src in src_HSQC:
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

def prepare_COSY_data_from_src_2(src_COSY_list):
    """
    Processes and scales HSQC spectral data from the source list.
    """
    processed_COSY = []

    for src in src_COSY_list:
        #for src in src_HSQC:

            # Filter out rows where both elements are not zero
            non_zero_mask = (src != 0).all(dim=1)
            filtered_src = src[non_zero_mask]

            if filtered_src.nelement() != 0:  # Check if tensor is not empty
                scaled_tensors = [filtered_src[:, 0] * 10, filtered_src[:, 1] * 10]
                combined_tensor = torch.stack(scaled_tensors, dim=1)
                processed_COSY.append(combined_tensor)
            else:
                processed_COSY.append(torch.tensor([]))  # Append an empty tensor for consistency

    return processed_COSY

def generate_df_for_HSQC_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_HSQC_list):
    """
    Generates a DataFrame containing successfully generated SMILES along with their corresponding target SMILES
    and a unique sample identifier. It also filters and returns new lists for source HSQC data and failed SMILES pairs.
    """
    succ_gen_list =[]
    src_HSQC_list_new = []
    failed_list = []
    for i, (gen_smi, trg_smi, src_HSQC) in enumerate(zip(gen_conv_SMI_list, trg_conv_SMI_list, src_HSQC_list)):
            mol = Chem.MolFromSmiles(gen_smi)
            if mol is not None:
                ran_num = random.randint(0, 100000)
                sample_id = f"{i}_{ran_num}"
                succ_gen_list.append([sample_id, gen_smi, trg_smi])
                src_HSQC_list_new.append(src_HSQC)
            else:
                failed_list.append([gen_smi,trg_smi])
                continue

    # Create a DataFrame of successful SMILES generations
    df_succ_smis = pd.DataFrame(succ_gen_list, columns=['sample-id', 'SMILES', "trg_SMILES"])
    return df_succ_smis, src_HSQC_list_new, failed_list

def generate_df_for_COSY_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_COSY_list):
    """
    Generates a DataFrame containing successfully generated SMILES along with their corresponding target SMILES
    and a unique sample identifier. It also filters and returns new lists for source HSQC data and failed SMILES pairs.
    """
    succ_gen_list =[]
    src_COSY_list_new = []
    failed_list = []
    for i, (gen_smi, trg_smi, src_HSQC) in enumerate(zip(gen_conv_SMI_list, trg_conv_SMI_list, src_COSY_list)):
            mol = Chem.MolFromSmiles(gen_smi)
            if mol is not None:
                ran_num = random.randint(0, 100000)
                sample_id = f"{i}_{ran_num}"
                succ_gen_list.append([sample_id, gen_smi, trg_smi])
                src_COSY_list_new.append(src_HSQC)
            else:
                failed_list.append([gen_smi,trg_smi])
                continue

    # Create a DataFrame of successful SMILES generations
    df_succ_smis = pd.DataFrame(succ_gen_list, columns=['sample-id', 'SMILES', "trg_SMILES"])
    return df_succ_smis, src_COSY_list_new, failed_list


def calculate_corr_max_prob(config, model_MMT, val_dataloader, stoi, itos):
    """
    Calculates and aggregates the probabilities of correct token predictions and maximum probabilities
    across all batches in a validation dataloader using a given model.
    """
    aggregated_corr_prob_multi, aggregated_corr_prob_avg, aggregated_max_prob_multi, aggregated_max_prob_avg =[],[],[],[]
    prob_dict_results = {}
    for idx, data_dict in enumerate(tqdm(val_dataloader)):

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT, data_dict, config) 
        trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob = predict_prop_correct_max_sequence_3(model_MMT, stoi, memory, src_padding_mask, trg_enc_SMI, config)

        # Ensure we're working with 2D tensors even for single molecules
        if corr_token_prob.dim() == 1:
            corr_token_prob = corr_token_prob.unsqueeze(1)
            max_token_prob = max_token_prob.unsqueeze(1)
        if isinstance(trg_tensor, int):
            trg_tensor = [trg_tensor]
        if isinstance(trg_tensor_max, int):
            trg_tensor_max = [trg_tensor_max]

        for corr_prob_list, max_prob_list, token_list, token_list_max in zip(corr_token_prob.T, max_token_prob.T, trg_tensor, trg_tensor_max):
            seq_corr_probs, seq_max_probs  = [], []
            for corr_prob, max_prob, token in zip(corr_prob_list, max_prob_list, token_list):                                                  
                if token == stoi["<EOS>"]:  # End of sequence
                    break
                seq_corr_probs.append(corr_prob.detach().item())
                seq_max_probs.append(max_prob.detach().item())
                    # Populate dictionaries with calculated probabilities

            prob_corr_multi = reduce(operator.mul, seq_corr_probs, 1)
            prob_corr_avg = statistics.mean(seq_corr_probs) if seq_corr_probs else 0
            prob_max_multi = reduce(operator.mul, seq_max_probs, 1)
            prob_max_avg = statistics.mean(seq_max_probs) if seq_max_probs else 0

            aggregated_corr_prob_multi.append(prob_corr_multi)
            aggregated_corr_prob_avg.append(prob_corr_avg)  
            aggregated_max_prob_multi.append(prob_max_multi)
            aggregated_max_prob_avg.append(prob_max_avg) 
        #import IPython; IPython.embed();


    prob_dict_results["aggregated_corr_prob_multi"] = aggregated_corr_prob_multi
    prob_dict_results["aggregated_corr_prob_avg"] = aggregated_corr_prob_avg
    prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
    prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
    return prob_dict_results



# def calculate_max_prob(config, model_MMT, val_dataloader, stoi, itos):
#     aggregated_corr_prob_multi, aggregated_corr_prob_avg, aggregated_max_prob_multi, aggregated_max_prob_avg =[],[],[],[]
#     prob_dict_results = {}
#     for idx, data_dict in enumerate(tqdm(val_dataloader)):
#         memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT, data_dict, config)
#         trg_tensor_max, max_token_prob = predict_prop_max_sequence_3(model_MMT, stoi, memory, src_padding_mask, trg_enc_SMI, config)

#         for max_prob_list, token_list_max in zip(max_token_prob.T, trg_tensor_max):
#             seq_max_probs  = []
#             for max_prob, token in zip(max_prob_list, token_list_max):                                                  
#                 if token == stoi["<EOS>"]:  # End of sequence
#                     break
#                 seq_max_probs.append(max_prob.detach().item())
#                     # Populate dictionaries with calculated probabilities

#             prob_max_multi = reduce(operator.mul, seq_max_probs, 1)
#             prob_max_avg = statistics.mean(seq_max_probs) if seq_max_probs else 0

#             aggregated_max_prob_multi.append(prob_max_multi)
#             aggregated_max_prob_avg.append(prob_max_avg) 
#         #import IPython; IPython.embed();

#     prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
#     prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
#     return prob_dict_results


# def predict_prop_max_sequence_3(model, stoi, memory, src_padding_mask, trg_enc_SMI, config):
#     """
#     Predicts the properties of each token in a sequence generated by a transformer model.

#     Parameters:
#     model (torch.nn.Module): The transformer model used for prediction.
#     stoi (dict): Dictionary mapping tokens to indices.
#     memory (torch.Tensor): The memory tensor output from the transformer's encoder.
#     src_padding_mask (torch.Tensor): The source padding mask.
#     trg_enc_SMI (torch.Tensor): The target encoded Simplified Molecular Input Line Entry System (SMILES).
#     gen_num (int): Number of generations for multinomial sampling.
#     config (obj): Configuration object containing model and operation settings.

#     Returns:
#     tuple: Contains tensors for target sequence, correct token probabilities, 
#            maximum probability sequence, maximum token probabilities, 
#            and multinomial token probabilities.
#     """

#     # Ensure the model is in evaluation mode
#     model.eval()

#     # Define the initial target tensor with <SOS> tokens
#     N = memory.size(1)
#     trg_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
#     trg_tensor_max = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)

#     # Token probabilities containers
#     corr_token_prob, max_token_prob = [], []

#     # Transpose target encoded SMILES and remove <EOS> token
#     trg_enc_SMI_T = trg_enc_SMI.transpose(0, 1)
#     real_trg = trg_enc_SMI_T[1:, :]
    
#     # Iterate over each token in the target sequence
#     with torch.no_grad():
#         for idx in range(real_trg.shape[0]):
#             # Prepare input for the decoder
#             gen_seq_length, N = trg_tensor_max.shape
#             gen_positions = torch.arange(gen_seq_length).unsqueeze(1).expand(gen_seq_length, N).to(config.device)
#             embedding_gen = model.dropout2(model.embed_trg(trg_tensor) + model.pe_trg(gen_positions))
#             gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

#             # Generate output from the decoder
#             output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
#             output = model.fc_out(output)
#             probabilities = F.softmax(output / config.temperature, dim=2)

#             # Process token probabilities
#             next_word = torch.argmax(probabilities[-1], dim=1)
#             max_prob = probabilities[-1].gather(1, next_word.unsqueeze(-1)).squeeze()
#             max_token_prob.append(max_prob)

#             # Update target tensor with max probability token
#             trg_tensor_max = torch.cat((trg_tensor_max, next_word.unsqueeze(0)), dim=0)

#     #import IPython; IPython.embed();
#     # Organize and return probabilities
#     max_token_prob = torch.stack(max_token_prob)#.transpose(0, 1)
#     #import IPython; IPython.embed();

#     # Remove <SOS> token from target sequences
#     trg_tensor_max = trg_tensor_max.transpose(0, 1)[:, 1:]
#     return trg_tensor_max, max_token_prob


def evaluate_greedy_2(model, stoi, itos, val_dataloader, config, randomize=False):
    """
    Evaluates the greedy generation approach over a dataset.
    """    
    gen_conv_SMI_list = []
    trg_conv_SMI_list = []
    src_HSQC_list = []
    src_COSY_list = []
    token_probs_list = []
    data_dict_list = []
    # generate all the smiles of trg and greedy gen
    for i, data_dict in tqdm(enumerate(val_dataloader)):
        #import IPython; IPython.embed();
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model,
                                                                       data_dict, 
                                                                       config)
        #print("eval_greedy")
        #print(src_HSQC)
        greedy_tensor, greedy_token_prob = greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config)
        gen_conv_SMI = hf.tensor_to_smiles(greedy_tensor, itos)
        #gen_conv_SMI, token_probs = ttf.tensor_to_smiles_and_prob(greedy_tensor.squeeze(1), greedy_token_prob, itos)
        token_probs_list.append(greedy_token_prob)
        gen_conv_SMI_list.extend(gen_conv_SMI)
        #gen_conv_SMI_list = gen_conv_SMI_list + gen_conv_SMI
        
        trg_enc_SMI = data_dict["trg_enc_SMI"]
        trg_enc_SMI = trg_enc_SMI.transpose(0, 1)
        trg_SMI_input = trg_enc_SMI[1:, :] # Remove <EOS> token from target sequence
        trg_conv_SMI = hf.tensor_to_smiles(trg_SMI_input, itos)
        trg_conv_SMI_list = trg_conv_SMI_list + trg_conv_SMI
        src_HSQC_list.extend(src_HSQC)
        src_COSY_list.extend(src_COSY)
        data_dict_list.append(data_dict)
    # Calculate validity of gen smiles
    validity_term = hf.get_validity_term(gen_conv_SMI_list) 
    # Calculate tanimoto similarity
    if randomize == True:
        random.shuffle(gen_conv_SMI_list)

    tanimoto_mean, tanimoto_std_dev, failed_pairs, tanimoto_scores_, tanimoto_scores_all, gen_conv_SMI_list_, trg_conv_SMI_list_, idx_list = hf.calculate_tanimoto_similarity_2(gen_conv_SMI_list, trg_conv_SMI_list)

    results_dict = {
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'gen_conv_SMI_list_': gen_conv_SMI_list_,
            'trg_conv_SMI_list_': trg_conv_SMI_list_,            
            'idx_list': idx_list,
            'token_probs_list': token_probs_list,
            'validity_term': validity_term,
            'tanimoto_scores_': tanimoto_scores_,
            'tanimoto_scores_all': tanimoto_scores_all,
            'data_dict_list': data_dict_list,
            'failed':failed_pairs}
    return results_dict, src_HSQC_list, src_COSY_list    



def predict_prop_correct_max_sequence_3(model, stoi, memory, src_padding_mask, trg_enc_SMI, config):
    """
    Predicts the properties of each token in a sequence generated by a transformer model.
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
            if idx <= real_trg.shape[0]:
                corr_probability = probabilities[-1].gather(1, real_trg[idx].unsqueeze(-1)).squeeze()
                corr_token_prob.append(corr_probability)

            # Update target tensor with actual next token
            next_word = real_trg[idx].unsqueeze(0)
            trg_tensor = torch.cat((trg_tensor, next_word), dim=0)

    #import IPython; IPython.embed();
    # Organize and return probabilities
    max_token_prob = torch.stack(max_token_prob)#.transpose(0, 1)
    corr_token_prob = torch.stack(corr_token_prob)#.transpose(0, 1)
    #import IPython; IPython.embed();

    # Remove <SOS> token from target sequences
    trg_tensor = trg_tensor.transpose(0, 1)[:, 1:]
    trg_tensor_max = trg_tensor_max.transpose(0, 1)[:, 1:]
    return trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob


def run_model_analysis(config, model_MMT, val_dataloader, stoi, itos):
    #import IPython; IPython.embed();
    print("calculate_corr_max_prob")

    prob_dict_results = calculate_corr_max_prob(config, model_MMT, val_dataloader, stoi, itos)
    try:
        final_prob_max_multi_sum = sum(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_multi_avg = statistics.mean(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_avg_avg = statistics.mean(prob_dict_results["aggregated_max_prob_avg"])
        final_prob_corr_multi_sum = sum(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_multi_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_avg_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_avg"])
    except:
        print("failed statistics")

    print("evaluate_greedy_2")
    results_dict, src_HSQC_list, src_COSY_list = evaluate_greedy_2(model_MMT, stoi, itos, val_dataloader, config, randomize=False)

    trg_conv_SMI_list = results_dict["trg_conv_SMI_list"]
    gen_conv_SMI_list = results_dict["gen_conv_SMI_list"]

    print("generate_df_for_HSQC_calculations")
    df_succ_smis, src_HSQC_list, failed_list = generate_df_for_HSQC_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_HSQC_list)
    df_succ_smis, src_COSY_list, failed_list = generate_df_for_COSY_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_COSY_list)
    tensor_HSQC = prepare_HSQC_data_from_src_2(src_HSQC_list)
    tensor_COSY = prepare_COSY_data_from_src_2(src_COSY_list)

    print("run_sgnn_sim_calculations_if_possible_return_spectra")
    #sgnn_avg_sim_error, HSQC_sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, vgmmt.sgnn_means_stds, config)
    avg_sim_error_HSQC, avg_sim_error_COSY, HSQC_sim_error_list, COSY_sim_error_list, batch_data = ttf.run_sgnn_sim_calculations_if_possible_return_spectra(df_succ_smis, tensor_HSQC, tensor_COSY, vgmmt.sgnn_means_stds, config)

    results_dict["HSQC_sim_error_list"] = HSQC_sim_error_list
    results_dict["COSY_sim_error_list"] = COSY_sim_error_list
    results_dict["batch_data"] = batch_data
    results_dict["df_succ_smis"] = df_succ_smis
    results_dict["tensor_HSQC"] = tensor_HSQC
    results_dict["tensor_COSY"] = tensor_COSY
    return prob_dict_results, results_dict



def load_data(config, stoi, stoi_MF, single=True, mode="val"):
    """Loads the dataset and Multimodal Transformer (MMT) model."""

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
    return dataloader



def load_MMT_model(config):
    """Loads the dataset and Multimodal Transformer (MMT) model."""
    # Initialize and load the multi-GPU model

    multi_gpu_model = TransformerMultiGPU(config)
    multi_gpu_model = multi_gpu_model.load_from_checkpoint(config.checkpoint_path, config=config)
    multi_gpu_model.model.to("cuda")

    return multi_gpu_model.model



def load_CLIP_model(config):

    CLIP_multi_gpu_model = CLIPMultiGPU(config)
    checkpoint_path = config.CLIP_model_path
    CLIP_model = CLIP_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)

    #CLIP_model, optimizer = CLIP_make(config, stoi, stoi_MF, itos)
    CLIP_model.to(config.device)

    return CLIP_model.CLIP_model



# def load_BLIP_model(config):

#     BLIP_multi_gpu_model = BLIPMultiGPU(config)
#     checkpoint_path = config.BLIP_model_path
#     BLIP_model = BLIP_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)

#     #CLIP_model, optimizer = CLIP_make(config, stoi, stoi_MF, itos)
#     BLIP_model.to(config.device)

#     return BLIP_model.BLIP_model


def run_test_mns_performance_CLIP_3(config, 
                                model_MMT,
                                model_CLIP,
                                val_dataloader,                                
                                 stoi, 
                                 itos, 
                                 MW_filter):
    ### Same code as function: run_multinomial_sampling
    n_times = config.multinom_runs
    results_dict = {} #defaultdict(list)
    temperature_orig = config.temperature
    for idx, data_dict in enumerate(val_dataloader):
        if idx % 10 == 0:
            print(idx)
        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        # to confirm that this smies is valid
        if Chem.MolFromSmiles(trg_conv_SMI) == None:
            continue
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                                    data_dict_dup, 
                                                                                    config)
        counter = 1
        while len(gen_conv_SMI_list)<n_times:
            # increase the temperature if not enough different molecules get generated               
            print(counter, len(gen_conv_SMI_list), config.temperature)
            if counter%20==0:
                print(trg_conv_SMI)
                break
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            #gen_conv_SMI_list = list(set(gen_conv_SMI_list)) ### for 10.000 commented out
            counter += 1
            config.temperature = config.temperature + 0.1  
        config.temperature = temperature_orig
        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]
        data_dict_dup_CLIP = rbgvm.duplicate_dict(data_dict, len(gen_conv_SMI_list))
        if len(gen_conv_SMI_list) != 0:
            mean_loss, losses, logits, targets, dot_similarity= model_CLIP.inference(data_dict_dup_CLIP, 
                                                                                gen_conv_SMI_list)

            combined_list = [[smile, num.item(), dot_sim.item(), prob] for smile, num, dot_sim, prob in zip(gen_conv_SMI_list, losses, dot_similarity, prob_list)]
            ### Sort by the lowest similarity
            #sorted_list = sorted(combined_list, key=lambda x: x[1])

            combined_list, failed_combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
            combined_list, batch_data = rbgvm.add_HSQC_COSY_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.multinom_runs) # config.MMT_batch

            sorted_list = sorted(combined_list, key=lambda x: -x[4]) # SMILES = 0, losses =1, dot_sim= 2, propb = 3, tanimoto = 4

            results_dict[trg_conv_SMI] = [sorted_list, batch_data]
        else:
            results_dict[trg_conv_SMI] = [None, None]

    return results_dict

def run_test_performance_CLIP_greedy_3(config, 
                                 model_MMT,
                                 model_CLIP,
                                 val_dataloader,                                        
                                 stoi, 
                                 stoi_MF, 
                                 itos, 
                                 itos_MF):


    results_dict = defaultdict(list)
    gen_conv_SMI_list = []
    token_probs_list = []
    src_HSQC_list = []
    failed = []

    # generate all the smiles of trg and greedy gen
    for idx, data_dict in enumerate(val_dataloader):
        if idx % 10 == 0:
            print(idx)
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 1)  # Maybe should hardcode it here as 64 - it will always cut it down to the number needed with ntimes

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                        data_dict_dup, 
                                                                        config)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        greedy_tensor, greedy_token_prob = greedy_sequence_2(model_MMT, stoi, itos, memory, src_padding_mask, config)
        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor.squeeze(1), greedy_token_prob, itos)
        tan_sim = try_calculate_tanimoto_from_two_smiles(trg_conv_SMI, gen_conv_SMI, 512, extra_info = False)
        mean_loss, losses, logits, targets, dot_similarity = model_CLIP.inference(data_dict_dup, gen_conv_SMI)

        sgnn_avg_sim_error, sim_error_list, batch_data = rbgvm.calculate_HSQC_error(config, data_dict_dup, gen_conv_SMI)
        combined_list =[gen_conv_SMI, losses.item(), dot_similarity.item(), tan_sim, sgnn_avg_sim_error]
        if tan_sim == None:
            failed.append([trg_conv_SMI, gen_conv_SMI, combined_list, batch_data])
            continue
        else:
            results_dict[trg_conv_SMI] = [combined_list, batch_data]
    #if i == config.n_samples:
    #    break
    #import IPython; IPython.embed();

    return results_dict, failed

def predict_corr_max_performance_metric(trg_tensor, corr_token_prob, max_token_prob, stoi):
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
    sample_prob_list_corr, sample_prob_list_max = [], []
    prob_corr_multi, prob_corr_avg, prob_max_multi, prob_max_avg = [], [], [], []

    # Iterate over each sequence to calculate probabilities
    seq_corr_probs, seq_max_probs, seq_multinom_probs = [], [], []

    for corr_prob, max_prob, token in zip(corr_token_prob, max_token_prob, trg_tensor[0]):
        # Initialize lists for individual sequence probabilities
        #import IPython; IPython.embed();
        # Iterate over each token in the sequence
        #for idx, (corr_prob, max_prob, multinom_prob, token) in enumerate(zip(corr_probs, max_probs, multinom_probs, tokens)):
        if token == stoi["<EOS>"]:  # End of sequence
            break
        seq_corr_probs.append(corr_prob.item())
        seq_max_probs.append(max_prob.item())

    # Calculate and append aggregated probabilities
    prob_corr_multi = reduce(operator.mul, seq_corr_probs, 1)
    prob_corr_avg = statistics.mean(seq_corr_probs) if seq_corr_probs else 0
    prob_max_multi = reduce(operator.mul, seq_max_probs, 1)
    prob_max_avg = statistics.mean(seq_max_probs) if seq_max_probs else 0

    # Populate dictionaries with calculated probabilities
    sample_dict = {
        "sample_prob_list_corr": seq_corr_probs,
        "sample_prob_list_max": seq_max_probs    
        }

    prop_dict = {
        "prob_corr_multi": prob_corr_multi,
        "prob_corr_avg": prob_corr_avg,
        "prob_max_multi": prob_max_multi,
        "prob_max_avg": prob_max_avg
        }

    return prop_dict, sample_dict

## There is something wrong with the predict_corr_max_performance_metric function
def calculate_corr_max_prob_2(config, model, stoi, val_dataloader, gen_num):
    
    prob_dict_results = {}
    aggregated_corr_prob_multi = []
    aggregated_corr_prob_avg = []
    aggregated_max_prob_multi = []
    aggregated_max_prob_avg = []
    #sample_dict = {}
    #for _ in range(2):  # Num_Runs is the number of times you want to run the entire process for randomized smiles

    for idx, data_dict in enumerate(tqdm(val_dataloader)):
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model, data_dict, config) 
        trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob = predict_prop_correct_max_sequence_3(model, stoi, memory, src_padding_mask, trg_enc_SMI, config)
        #import IPython; IPython.embed();

        prop_dict, sample_dict = predict_corr_max_performance_metric(trg_tensor, corr_token_prob, max_token_prob, stoi)
        aggregated_corr_prob_multi.append(prop_dict["prob_corr_multi"])
        aggregated_corr_prob_avg.append(prop_dict["prob_corr_avg"])  
        aggregated_max_prob_multi.append(prop_dict["prob_max_multi"])
        aggregated_max_prob_avg.append(prop_dict["prob_max_avg"]) 
    #import IPython; IPython.embed();

    prob_dict_results["aggregated_corr_prob_multi"] = aggregated_corr_prob_multi
    prob_dict_results["aggregated_corr_prob_avg"] = aggregated_corr_prob_avg
    prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
    prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
    #print("calculate_corr_max_prob_2")
    #import IPython; IPython.embed();

    return prob_dict_results, sample_dict


def run_test_performance_CLIP_3(config, 
                                model_MMT,
                                val_dataloader,                                   
                                stoi):


    total_results = {}
    # Number of times to duplicate each tensor 
    gen_num = 1 # Multinomial sampling number
    prob_dict_results, _ = calculate_corr_max_prob_2(config, model_MMT, stoi, val_dataloader, gen_num)
    try:
        final_prob_max_multi_sum = sum(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_multi_avg = statistics.mean(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_avg_avg = statistics.mean(prob_dict_results["aggregated_max_prob_avg"])
        final_prob_corr_multi_sum = sum(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_multi_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_avg_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_avg"])
    except:
        print("failed statistics")
    total_results["statistics_multiplication_avg"] = [final_prob_corr_multi_avg,
                                                    final_prob_max_multi_avg]

    total_results["statistics_multiplication_sum"] = [final_prob_corr_multi_sum,
                                                    final_prob_max_multi_sum]

    total_results["statistics_avg_avg"] = [final_prob_corr_avg_avg,
                                            final_prob_max_avg_avg]
    return total_results


#__________________________________________________
### Sample 1000 unique molecules and plot their tanimoto similartyy 
### compared to the target
def run_multinomial_sampling(config, model_MMT, val_dataloader, itos, stoi, MW_filter=False):
    n_times = config.multinom_runs
    results_dict = defaultdict(list)
    temperature_orig = config.temperature
    for idx, data_dict in tqdm(enumerate(val_dataloader)):
        if idx % 10 ==0:
            print(idx)

        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                   data_dict_dup, 
                                                                   config)
        counter = 1
        while len(gen_conv_SMI_list)<n_times:
            # increase the temperature if not enough different molecules get generated               
            if counter%20==0:
                print(trg_conv_SMI)
                break
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            #gen_conv_SMI_list = list(set(gen_conv_SMI_list)) ### for 10.000 commented out
            counter += 1
            config.temperature = config.temperature + 0.1    
        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]

        tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
        results_dict[idx].append({
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'tanimoto_sim': tanimoto_list_all,
            'tanimoto_mean': tanimoto_mean,
            'tanimoto_std_dev': tanimoto_std_dev,
            'failed': failed,
            'prob_list': prob_list,
        })
        config.temperature = temperature_orig 
    return config, results_dict



# Multinomial
def multinomial_sequence_multi_2(model, memory, src_padding_mask, stoi, config):
    # Initialization

    model.eval()
    N = memory.size(1)
    multinom_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    multinom_token_prob = []  

    # Sequence Prediction
    with torch.no_grad():
        for idx in range(0, config.max_len):
            # [The same logic you already had]
            gen_seq_length, N = multinom_tensor.shape
            gen_positions = (
                torch.arange(0, gen_seq_length)
                .unsqueeze(1)
                .expand(gen_seq_length, N)
                .to(config.device))

            embedding_gen = model.dropout2((model.embed_trg(multinom_tensor) + model.pe_trg(gen_positions)))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)
            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
            output = model.fc_out(output)

            probabilities = F.softmax(output / config.temperature, dim=2)
            ## Capturing the probability of the next predicted token with multinomial sampling
            next_word = torch.multinomial(probabilities[-1, :, :], 1)  
            sel_prob = probabilities[-1, :, :].gather(1, next_word).squeeze()  # Get the probability of the predicted token
            multinom_token_prob.append(sel_prob)
            next_word = next_word.squeeze(1).unsqueeze(0)
            multinom_tensor = torch.cat((multinom_tensor, next_word), dim=0)
    # import IPython; IPython.embed();
    multinom_token_prob = torch.stack(multinom_token_prob) 

    # remove "SOS" token
    multinom_tensor = multinom_tensor[1:,:]
    multinom_token_prob = multinom_token_prob[1:,:]
    
    return multinom_tensor, multinom_token_prob


def run_greedy_sampling(config, model_MMT, val_dataloader, itos, stoi):

    results_dict = defaultdict(list)
    #model_MMT, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF, single=True, mode="val")

    gen_conv_SMI_list, trg_conv_SMI_list, prob_list, src_HSQC_list, src_COSY_list = [], [], [], [], []
    for idx, data_dict in enumerate(val_dataloader):
        trg_enc_SMI = data_dict["trg_enc_SMI"]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI.T[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                   data_dict, 
                                                                   config)

        greedy_tensor, greedy_token_prob = greedy_sequence_2(model_MMT, stoi, itos, memory, src_padding_mask, config)

        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor, greedy_token_prob.T, itos)

        gen_conv_SMI_list.extend(gen_conv_SMI)
        trg_conv_SMI_list.extend(trg_conv_SMI)
        prob_list.extend(token_probs)
        src_HSQC_list.extend(src_HSQC)
        src_COSY_list.extend(src_COSY)
        
    tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
    results_dict = {
        'gen_conv_SMI_list': gen_conv_SMI_list,
        'trg_conv_SMI_list': trg_conv_SMI_list,
        'tanimoto_sim': tanimoto_list_all,
        'tanimoto_mean': tanimoto_mean,
        'tanimoto_std_dev': tanimoto_std_dev,
        'failed': failed,
        'prob_list': prob_list,
        "src_HSQC_list": src_HSQC_list,
        "src_COSY_list": src_COSY_list,
        }
    
    return config, results_dict

    
# def greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config):
#     """
#     Generates a sequence of tokens using a greedy approach.

#     Parameters:
#     model (torch.nn.Module): The trained model for sequence generation.
#     stoi (dict): Mapping of tokens to indices.
#     itos (dict): Mapping of indices to tokens.
#     memory (Tensor): Memory tensor from the model.
#     src_padding_mask (Tensor): Source padding mask.
#     config (object): Configuration containing model parameters.

#     Returns:
#     tuple: Tensor of generated tokens and their probabilities.
#     """
#     model.eval()
#     N = memory.size(1)
#     greedy_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
#     greedy_token_prob = []

#     with torch.no_grad():
#         for _ in range(config.max_len):
#             gen_seq_length = greedy_tensor.size(0)
#             gen_positions = torch.arange(gen_seq_length, device=config.device).unsqueeze(1).expand(gen_seq_length, N)
#             embedding_gen = model.embed_trg(greedy_tensor) + model.pe_trg(gen_positions)
#             if model.training:
#                 embedding_gen = model.dropout2((embedding_gen))
#             gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

#            #import IPython; IPython.embed();
#             output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
           
#             output = model.fc_out(output)
#             probabilities = F.softmax(output / config.temperature, dim=2)

#             ## Capturing the probability of the best next predicted token
#             next_word = torch.argmax(probabilities[-1, :, :], dim=1)
#             max_prob = probabilities[-1, :, :].gather(1, next_word.unsqueeze(-1)).squeeze()
#             greedy_token_prob.append(max_prob)
#             next_word = next_word.unsqueeze(0)

#             greedy_tensor = torch.cat((greedy_tensor, next_word), dim=0)
#             if (next_word == 0).all():
#                 break
                


#     greedy_token_prob = torch.stack(greedy_token_prob) 
#     #greedy_token_prob = greedy_token_prob.transpose(0, 1)
#     # import IPython; IPython.embed();

#     # remove "SOS" token
#     greedy_tensor = greedy_tensor[1:,:]
#     greedy_token_prob = greedy_token_prob[1:,:]
    
#     return greedy_tensor, greedy_token_prob


# def greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config):
#     """
#     Generates a sequence of tokens using a greedy approach.

#     Parameters:
#     model (torch.nn.Module): The trained model for sequence generation.
#     stoi (dict): Mapping of tokens to indices.
#     itos (dict): Mapping of indices to tokens.
#     memory (Tensor): Memory tensor from the model.
#     src_padding_mask (Tensor): Source padding mask.
#     config (object): Configuration containing model parameters.

#     Returns:
#     tuple: Tensor of generated tokens and their probabilities.
#     """
#     model.eval()
#     N = memory.size(1)
#     greedy_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
#     greedy_token_prob = []

#     with torch.no_grad():
#         for _ in range(config.max_len):
#             gen_seq_length = greedy_tensor.size(0)
#             gen_positions = torch.arange(gen_seq_length, device=config.device).unsqueeze(1).expand(gen_seq_length, N)
#             embedding_gen = model.dropout2((model.embed_trg(greedy_tensor) + model.pe_trg(gen_positions)))
#             gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

#            #import IPython; IPython.embed();
#             output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
           
#             output = model.fc_out(output)
#             probabilities = F.softmax(output / config.temperature, dim=2)

#             ## Capturing the probability of the best next predicted token
#             next_word = torch.argmax(probabilities[-1, :, :], dim=1)
#             max_prob = probabilities[-1, :, :].gather(1, next_word.unsqueeze(-1)).squeeze()
#             greedy_token_prob.append(max_prob)
#             next_word = next_word.unsqueeze(0)

#             greedy_tensor = torch.cat((greedy_tensor, next_word), dim=0)
#             if (next_word == 0).all():
#                 break
                


#     greedy_token_prob = torch.stack(greedy_token_prob) 
#     #greedy_token_prob = greedy_token_prob.transpose(0, 1)

#     # remove "SOS" token
#     greedy_tensor = greedy_tensor[1:,:]
#     greedy_token_prob = greedy_token_prob[:,:]
    
#     return greedy_tensor, greedy_token_prob


def greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config):
    """
    Generates a sequence of tokens using a greedy approach.
    """
    model.eval()
    N = memory.size(1)  # Batch size
    greedy_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    greedy_token_prob = []

    with torch.no_grad():
        for _ in range(config.max_len):
            gen_seq_length = greedy_tensor.size(0)
            gen_positions = torch.arange(gen_seq_length, device=config.device).unsqueeze(1).expand(gen_seq_length, N)
            embedding_gen = model.embed_trg(greedy_tensor) + model.pe_trg(gen_positions)
            if model.training:
                embedding_gen = model.dropout2((embedding_gen))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
           
            output = model.fc_out(output)
            probabilities = F.softmax(output / config.temperature, dim=2)

            next_word = torch.argmax(probabilities[-1, :, :], dim=1)
            max_prob = probabilities[-1, :, :].gather(1, next_word.unsqueeze(-1)).squeeze()
            greedy_token_prob.append(max_prob)
            next_word = next_word.unsqueeze(0)

            greedy_tensor = torch.cat((greedy_tensor, next_word), dim=0)
            if (next_word == 0).all():
                break

    greedy_token_prob = torch.stack(greedy_token_prob)
    #import IPython; IPython.embed();

    # Remove "SOS" token
    greedy_tensor = greedy_tensor[1:]
    greedy_token_prob = greedy_token_prob


    # Handle single molecule case
    if N == 1:
        greedy_tensor = greedy_tensor#.squeeze(1)
        greedy_token_prob = greedy_token_prob.unsqueeze(-1)
    # else:
    #     greedy_tensor = greedy_tensor.transpose(0, 1)
    #     greedy_token_prob = greedy_token_prob.transpose(0, 1)
    
    return greedy_tensor, greedy_token_prob


def deduplicate_smiles(smiles_list, prob_list):
    # Create a dictionary to hold unique smiles and their corresponding probabilities
    unique_smiles = {}

    # Loop over the SMILES and their corresponding probabilities
    for smi, prob in zip(smiles_list, prob_list):
        if smi not in unique_smiles:
            unique_smiles[smi] = prob

    # Extracting the deduplicated lists
    deduped_smiles = list(unique_smiles.keys())
    deduped_probs = list(unique_smiles.values())

    return deduped_smiles, deduped_probs

# Function to filter valid SMILES
def filter_probs_and_valid_smiles_and_canonicolize(smiles_list, token_probs, canonical=True, isomericSmiles=False):
    valid_smiles = []
    valid_token_probs = []
    for smi, prob in zip(smiles_list, token_probs):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            gen_smi = Chem.MolToSmiles(mol, canonical=canonical, doRandom=False, isomericSmiles=isomericSmiles)
            valid_smiles.append(gen_smi)
            valid_token_probs.append(prob)
    return valid_smiles, valid_token_probs


# Function to calculate the rounded molecular weight
def calc_rounded_mw(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return round(Descriptors.MolWt(mol))
    except:
        return None



def filter_for_MW(trg_conv_SMI, gen_conv_SMI):

    # Calculate the rounded molecular weight of the target molecule
    trg_mw = calc_rounded_mw(trg_conv_SMI)

    # Filter the list based on molecular weight
    filtered_gen_smis = [smi for smi in gen_conv_SMI if calc_rounded_mw(smi) == trg_mw]  
    return filtered_gen_smis


def filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, prob_list):

    # Calculate the rounded molecular weight of the target molecule
    trg_mw = calc_rounded_mw(trg_conv_SMI)

    # Filter the list based on molecular weight
    filtered_gen_smis = [smi for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_rounded_mw(smi) == trg_mw]  
    filtered_prob_list = [token_prob for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_rounded_mw(smi) == trg_mw]  
    return filtered_gen_smis, filtered_prob_list

    
def filter_dict_results_for_MW(results_dict):
    tanimoto_values = []
    gen_trg_smi_lists = []
    for i, idx in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx

        for result in results_dict[idx]:
            trg_smi = result["trg_conv_SMI_list"][0]
            smi_list = result["gen_conv_SMI_list"]

            # Calculate the rounded molecular weight of the target molecule
            trg_mw = calc_rounded_mw(trg_smi)

            # Filter the list based on molecular weight
            filtered_gen_smis = [smi for smi in smi_list if calc_rounded_mw(smi) == trg_mw]  
            trg_conv_SMI_list = [trg_smi for i in range(len(filtered_gen_smis))]
            gen_trg_smi_lists.append([filtered_gen_smis, trg_conv_SMI_list])
            tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(filtered_gen_smis, trg_conv_SMI_list)
            tanimoto_values.append(tanimoto_list_all)
    return tanimoto_values, gen_trg_smi_lists


def plot_hist_MN_sampling(config, results_dict):
    # Number of unique idx values
    num_idx = len(results_dict)

    # Create a figure with subplots (if there are many idx, you might need to adjust the size and layout)
    plt.figure(figsize=(10, 6))

    for i, idx in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx
        tanimoto_values = []
        for result in results_dict[idx]:
            tanimoto_values.extend(result['tanimoto_sim'])
        #tanimoto_values = [val for result in results_dict[idx] for val in result['tanimoto_sim'] if val != 0]

        # Create a subplot for each idx
       
        plt.hist(tanimoto_values, bins=100+idx*3, alpha=0.7, label=f'Idx {idx}, Nr: {len(tanimoto_values)}')
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Tanimoto Similarity for {idx+1} Molecule generations')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    
def plot_hist_MN_sampling_all(config, results_dict):
    # Number of unique idx values
    num_idx = len(results_dict)

    # Create a figure with subplots (if there are many idx, you might need to adjust the size and layout)
    plt.figure(figsize=(10, 6))
    tanimoto_values = []
    for i, key in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx
        for result in results_dict[key]:
            tanimoto_values.extend(result['tanimoto_sim'])
        #tanimoto_values = [val for result in results_dict[idx] for val in result['tanimoto_sim'] if val != 0]

        # Create a subplot for each idx
       
    plt.hist(tanimoto_values, bins=100+idx*3, alpha=0.7, label=f'Idx {idx}, Nr: {len(tanimoto_values)}')
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Tanimoto Similarity for {i+1} Molecule generations')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_hist_MN_sampling_filtered(config, results_dict):
    ##filter out all molecules with the same MW as the target

    # Number of unique idx values
    num_idx = len(results_dict)
    tanimoto_values, gen_trg_smi_lists = filter_dict_results_for_MW(results_dict)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))

    # Plot each list in the histogram
    for idx, lst in enumerate(tanimoto_values):
        
        plt.hist(lst, bins=20, alpha=0.5, label=f'List {idx+1}, Nr: {len(lst)}')

    # Adding labels and title
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Tanimoto Similarity with same MW')
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()
    return tanimoto_values, gen_trg_smi_lists


def calc_percentage_top_x_correct(results_dict, top_x):
    count_yes = 0
    count_no = 0
    for i, idx in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx

        for result in results_dict[idx]:
            tanimoto_sim = result["tanimoto_sim"]
            tanimoto_sim = tanimoto_sim[:top_x]
            if 1 in tanimoto_sim:
                count_yes += 1
            else:
                count_no += 1
        percentage = count_yes/(count_yes +count_no)
    return percentage
    
def calc_percentage_top_x_correct_greedy(results_dict):
    count_yes = 0
    count_no = 0

    for i in results_dict["tanimoto_sim"]:
        if i==1:
            count_yes += 1
        else:
            count_no += 1

    percentage = count_yes/(count_yes + count_no)
    return percentage
    

def run_precentage_calculation(config, itos, stoi, stoi_MF, MW_filter):
    
    model_MMT = load_MMT_model(config)
    val_dataloader = load_data(config, stoi, stoi_MF, single=True, mode="val")  
    config.temperature = 1
    config, results_dict_mns_10 = run_multinomial_sampling(config, model_MMT, val_dataloader, itos, stoi, MW_filter=MW_filter)
    top_x = 10
    percentage_top_10 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 5
    percentage_top_5 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 3
    percentage_top_3 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 1
    percentage_top_1 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    
    val_dataloader_multi = load_data(config, stoi, stoi_MF, single=False, mode="val")  
    config, results_dict_greedy = run_greedy_sampling(config, model_MMT, val_dataloader_multi, itos, stoi)

    percentage_1_greedy = calc_percentage_top_x_correct_greedy(results_dict_greedy)

    percentage_collection = [percentage_1_greedy, percentage_top_1, percentage_top_3, percentage_top_5, percentage_top_10]

    return percentage_collection, results_dict_mns_10, results_dict_greedy

    
def add_tanimoto_similarity(trg_conv_SMI, combined_list):
    # Generate fingerprint for the ground truth molecule
    ground_truth_mol = Chem.MolFromSmiles(trg_conv_SMI)
    ground_truth_fp = AllChem.GetMorganFingerprintAsBitVect(ground_truth_mol, 2, nBits=512)
    # Calculate Tanimoto similarity and add to the list
    new_combined_list = []
    failed_combined_list = []
    for item in combined_list:

        smiles = item[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Check if the molecule is valid
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(ground_truth_fp, fp)
            item.append(tanimoto_similarity)
            new_combined_list.append(item)
        else:
            item.append(None)
            failed_combined_list.append(item)
    return new_combined_list, failed_combined_list
#print(combined_list)


def try_calculate_tanimoto_from_two_smiles(smi1, smi2, nbits, extra_info = False):
    """This function takes two smile_stings and 
    calculates the Tanimoto similarity and returns it and prints it out"""
    
    try:
        pattern1 = Chem.MolFromSmiles(smi1)
        pattern2 = Chem.MolFromSmiles(smi2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(pattern2, 2, nBits=nbits)

        tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        tan_sim = round(tan_sim,4)
        if extra_info:
            print(f"Smiles 1: {smi1} \n Target Smiles: {smi2} \nTanimoto score:{tan_sim}")

        return tan_sim
    except:
        return None

from collections import defaultdict

def calculate_tanimoto_of_all_compared_to_trg(trg_conv_SMI, gen_conv_SMI_list):
    tani_list = []
    ground_truth_mol = Chem.MolFromSmiles(trg_conv_SMI)
    ground_truth_fp = AllChem.GetMorganFingerprintAsBitVect(ground_truth_mol, 2, nBits=512)
    for smi in gen_conv_SMI_list:
        #smiles = item[0]
        mol = Chem.MolFromSmiles(smi)
        if mol:  # Check if the molecule is valid
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(ground_truth_fp, fp)
            tani_list.append(tanimoto_similarity)
    return tani_list


def run_mns_hit_counter_experiment(config, model_MMT, val_dataloader, itos, stoi, MW_filter, max_runs):
    percentage_collection = []
    molecule_tani_comparison_lists = []
    one_finder_list = []
    n_times = config.multinom_runs

    results_dict = defaultdict(list)
    # generate all the smiles of trg and greedy gen
    for i, data_dict in enumerate(val_dataloader):
        gen_conv_SMI_list = []
        gen_conv_SMI_list, trg_conv_SMI_list,  prob_list = [], [], []

        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        ### multiply the input to paralellize the generation
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                       data_dict_dup, 
                                                                       config)
        counter = 0
        ### Here I increase the temperature in case if it does not generate enough diverse molecules but sets it back to the original
        ### value after enough molecules were found.
        temp_orig = config.temperature
        while len(gen_conv_SMI_list)<n_times and counter<max_runs:
            # increase the temperature if not enough different molecules get generated
            if counter %10 == 0:
                print(counter, len(gen_conv_SMI_list), config.temperature )
                
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            
            tani_list = calculate_tanimoto_of_all_compared_to_trg(trg_conv_SMI, gen_conv_SMI_list)
            if 1 in tani_list:
                break
            #print(counter, len(gen_conv_SMI_list))
            counter += 1
            config.temperature = config.temperature + 0.1    
            
        try:
            one_finder_list.append(tani_list.index(1.0)+1)  # because indices start with 0
        except:
            one_finder_list.append(-5)
        trg_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]
        molecule_tani_comparison_lists.append([gen_conv_SMI_list,trg_SMI_list, tani_list])
        config.temperature = temp_orig 
        #break
    return one_finder_list, molecule_tani_comparison_lists

def sel_data_slice_and_save_as_csv(config):
    """ Saves the selected molecules in a new csv and replaces the csv_SMI_targets"""
    # File path
    file_path = config.csv_path_val  # Replace with your file path

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select the first X rows
    df_selected = df.head(config.data_size)

    # Save the selected rows to a new CSV file
    new_file_path = file_path.replace('.csv', f'_sel_{config.data_size}.csv')
    df_selected.to_csv(new_file_path, index=False)
    config.csv_SMI_targets = new_file_path
    return config



def filter_smiles(df, smi_list):
    """
    Filter out SMILES strings from smi_list that are present in the DataFrame df.

    Parameters:
    df (pandas.DataFrame): DataFrame with a 'SMILES' column.
    smi_list (list): List of SMILES strings.

    Returns:
    list: Filtered list of SMILES strings.
    """
    df_smiles_set = set(df['SMILES'])
    filtered_list = [smiles for smiles in smi_list if smiles not in df_smiles_set]
    return filtered_list


import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt

def analyze_and_plot(results_dict, mode):
    ranks = []
    failed = []
    
    for key, value in results_dict.items():
        # Extracting second and third numbers from each sublist
        if mode == "HSQC_sim":
            similaritys = [item[4] for item in value]
        elif mode == "dot_sim":
            similaritys = [item[2] for item in value]
        tani_sims = [item[3] for item in value]
        
        if tani_sims[0] == 1:
            # First number
            first_number = similaritys[0]

            # Sorting the numbers in descending order
            if mode == "HSQC_sim":
                sorted_numbers = sorted(similaritys, reverse=False)
            elif mode == "dot_sim":
                sorted_numbers = sorted(similaritys, reverse=True)

            # Finding the rank of the first number
            rank_of_first_number = sorted_numbers.index(first_number) + 1  # Adding 1 because index starts from 0

            ranks.append(rank_of_first_number)
        else:
            failed.append([key, value])
            #print(third_numbers)
            
    # Plotting the histogram of ranks
    plt.hist(ranks, bins=range(1, len(value) + 2), align='left')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f'Rank Histogram of {mode} corresponding correct Molecule')
    plt.xticks(range(1, len(value) + 1))
    plt.show()
    return failed
    
    
# Function to plot a molecule with additional data
def plot_molecule_with_data(smiles, cosine_sim, tanimoto, HSQC_error):
    mol = Chem.MolFromSmiles(smiles)
    cosine_sim_rounded = round(float(cosine_sim), 3) if cosine_sim else ''
    tanimoto_rounded = round(float(tanimoto), 3) if tanimoto else ''
    HSQC_error_rounded = round(float(HSQC_error), 4) if HSQC_error else ''
    fig, ax = plt.subplots()
    img = Draw.MolToImage(mol, size=(300, 300))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"CLIP: {cosine_sim_rounded}, Tanimoto: {tanimoto_rounded},  HSQC_err: {HSQC_error_rounded}")
    plt.show()

    ### PLOT Molecules

def plot_CLIP_molecules(results_dict, investigate_number, stop_nr):
    #investigate_number = 5
    # Plot the key SMILES


    # Plot the first item of each list with the third and fourth elements
    for idx, data_val in enumerate(results_dict.values()):
        
        if idx == investigate_number:
            key_smiles = list(results_dict.keys())[investigate_number]   
            plot_molecule_with_data(key_smiles, '', '', '')
            
            for idy, lists in enumerate(data_val[0]):
                first_smiles, _, cosine_sim, tanimoto, HSQC_error = lists
                plot_molecule_with_data(first_smiles, cosine_sim, tanimoto, HSQC_error)
                if idy == stop_nr:
                    break
                    
def generate_mns_list(config, 
                  model_MMT, 
                  val_dataloader,
                  stoi, 
                  itos, 
                  MW_filter):
    ### Same code as function: run_multinomial_sampling
    n_times = config.multinom_runs
    gen_dict = {} #defaultdict(list)
    temperature_orig = config.temperature
    for idx, data_dict in enumerate(val_dataloader):
        if idx % 10 == 0:
            print(idx)
        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = ttf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                   data_dict_dup, 
                                                                   config)
        counter = 1
        while len(gen_conv_SMI_list)<n_times:
            # increase the temperature if not enough different molecules get generated               
            print(counter, len(gen_conv_SMI_list), config.temperature)
            if counter%30==0:
                print(trg_conv_SMI)
                break

            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)
            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            #gen_conv_SMI_list = list(set(gen_conv_SMI_list)) ### for 10.000 commented out
            counter += 1
            config.temperature = config.temperature + 0.1  
            #print(time.time() -start)
            #start = time.time()
        config.temperature = temperature_orig

        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]
        
        gen_dict[trg_conv_SMI] = [gen_conv_SMI_list, trg_conv_SMI_list, data_dict]
    return gen_dict

def run_clip_similarity_check(config, model_CLIP, gen_dict):
    results_dict ={}
    for trg_conv_SMI, values in gen_dict.items():
        gen_conv_SMI_list, trg_conv_SMI_list, data_dict = values
        data_dict_dup = rbgvm.duplicate_dict(data_dict, len(gen_conv_SMI_list))

        mean_loss, losses, logits, targets, dot_similarity= model_CLIP.inference(data_dict_dup, 
                                                                    gen_conv_SMI_list)

        combined_list = [[smile, num.item(), dot_sim.item()] for smile, num, dot_sim in zip(gen_conv_SMI_list, losses, dot_similarity)]
        ### Sort by the lowest similarity
        #sorted_list = sorted(combined_list, key=lambda x: x[1])

        combined_list, failed_combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
        #combined_list, batch_data = rbgvm.add_HSQC_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.multinom_runs) # config.MMT_batch
        #sorted_list = sorted(combined_list, key=lambda x: -x[3]) # SMILES = 0, losses =1, dot_sim= 2, tanimoto = 3 


        results_dict[trg_conv_SMI] = [combined_list]
    return results_dict

def run_HSQC_similarity_check(config, gen_dict, results_dict):
    results_dict_HSQC ={}
    for trg_conv_SMI, values in gen_dict.items():
        gen_conv_SMI_list, trg_conv_SMI_list, data_dict = values
        data_dict_dup = rbgvm.duplicate_dict(data_dict, len(gen_conv_SMI_list))

        src_HSQC_list = [data_dict_dup["src_HSQC"]]
        tensor_HSQC = rbgvm.prepare_HSQC_data_from_src(src_HSQC_list)

        sgnn_avg_sim_error, sim_error_list, batch_data = rbgvm.calculate_HSQC_error(config, data_dict_dup, gen_conv_SMI_list)
        results_dict_HSQC[trg_conv_SMI] = [gen_conv_SMI_list, sim_error_list]# batch_data]
    return results_dict_HSQC

def combine_CLIP_HSQC_data(results_dict_CLIP, results_dict_HSQC):

    final_results = {}
    for trg, data in results_dict_CLIP.items():
        for idx, data_list in enumerate(data[0]):
            try:
                hsqc_error = results_dict_HSQC[trg][1][idx]
                data[0][idx].append(hsqc_error)
            except:
                data[0][idx].append(-9)            
        data = sorted(data[0], key=lambda x: -x[3]) # SMILES = 0, losses =1, dot_sim= 2, tanimoto = 3 
        final_results[trg] = data
    return final_results


def filter_invalid_inputs(results_dict):
    """ Check if there is just a [None,None] entry from the run_test_mns_performance_CLIP_3 function"""
    filtered_dict = {}
    counter = 0
    for key, value in results_dict.items():
        # Assuming 'combined_list' is the first item in the list which is the value of the dictionary.
        # And we're checking if the first item in 'combined_list' is not '[None, None]'.
        if value[0] is not None:# and value[1] is not None:
            filtered_dict[key] = value
        else:
            counter+=1
    return filtered_dict, counter

def calc_molecular_formula(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return CalcMolFormula(mol)
    except:
        return None

def filter_for_MF_2(trg_conv_SMI, gen_conv_SMI, prob_list):
    # Calculate the molecular formula of the target molecule
    trg_mf = calc_molecular_formula(trg_conv_SMI)

    # Filter the list based on molecular formula
    filtered_gen_smis = [smi for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_molecular_formula(smi) == trg_mf]
    filtered_prob_list = [token_prob for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_molecular_formula(smi) == trg_mf]
    return filtered_gen_smis, filtered_prob_list


def run_multinomial_sampling_v2(config, model_MMT, val_dataloader, itos, stoi, MW_filter=False, MF_filter=False):
    n_times = config.multinom_runs
    results_dict = defaultdict(list)
    temperature_orig = config.temperature

    for idx, data_dict in tqdm(enumerate(val_dataloader)):
        if idx % 10 == 0:
            print(idx)

        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 16)

        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                   data_dict_dup, 
                                                                   config)
        counter = 1
        removed = 0
        while len(gen_conv_SMI_list) < n_times:
            # Increase the temperature if not enough different molecules get generated               
            if counter % 5 == 0:
                print(trg_conv_SMI)
                break
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # Filter valid SMILES and canonicalize
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)
            #print(f"valid smi: {len(set(gen_conv_SMI))}")
            #import IPython; IPython.embed();
            before_gen = len(gen_conv_SMI)
            if MW_filter:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
                #print(f"MW_filter: {before_gen - len(gen_conv_SMI)}")
                removed_1 = before_gen - len(gen_conv_SMI)
                removed += removed_1
            if MF_filter:
                gen_conv_SMI, token_probs = filter_for_MF_2(trg_conv_SMI, gen_conv_SMI, token_probs)
                #print(f"MF_filter: {before_gen - len(gen_conv_SMI)}")
                removed_2 = before_gen - len(gen_conv_SMI)
                removed += removed_2

            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            counter += 1
            config.temperature = config.temperature + 0.2
        #print(removed, len(gen_conv_SMI_list))
        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for _ in range(len(gen_conv_SMI_list))]

        tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
        results_dict[idx].append({
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'tanimoto_sim': tanimoto_list_all,
            'tanimoto_mean': tanimoto_mean,
            'tanimoto_std_dev': tanimoto_std_dev,
            'failed': failed,
            'prob_list': prob_list,
        })
        config.temperature = temperature_orig
    return config, results_dict

def run_precentage_calculation_v2(config, itos, stoi, stoi_MF, MW_filter=False, MF_filter=False):
    model_MMT = load_MMT_model(config)
    val_dataloader = load_data(config, stoi, stoi_MF, single=True, mode="val")  
    config.temperature = 1
    config, results_dict_mns_10 = run_multinomial_sampling_v2(config, model_MMT, val_dataloader, itos, stoi, MW_filter=MW_filter, MF_filter=MF_filter)
    top_x = 10
    percentage_top_10 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 5
    percentage_top_5 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 3
    percentage_top_3 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 1
    percentage_top_1 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    
    val_dataloader_multi = load_data(config, stoi, stoi_MF, single=False, mode="val")  
    config, results_dict_greedy = run_greedy_sampling_v2(config, model_MMT, val_dataloader_multi, itos, stoi, MW_filter=False, MF_filter=False)
    #config, results_dict_greedy = run_greedy_sampling(config, model_MMT, val_dataloader_multi, itos, stoi, MW_filter=MW_filter)

    percentage_1_greedy = calc_percentage_top_x_correct_greedy(results_dict_greedy)

    percentage_collection = [percentage_1_greedy, percentage_top_1, percentage_top_3, percentage_top_5, percentage_top_10]

    return percentage_collection, results_dict_mns_10, results_dict_greedy

def run_greedy_sampling_v2(config, model_MMT, val_dataloader, itos, stoi, MW_filter=False, MF_filter=False):
    results_dict = defaultdict(list)
    gen_conv_SMI_list, trg_conv_SMI_list, prob_list, src_HSQC_list, src_COSY_list = [], [], [], [], []
    for idx, data_dict in enumerate(val_dataloader):
        trg_enc_SMI = data_dict["trg_enc_SMI"]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI.T[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                   data_dict, 
                                                                   config)

        greedy_tensor, greedy_token_prob = greedy_sequence_2(model_MMT, stoi, itos, memory, src_padding_mask, config)

        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor, greedy_token_prob.T, itos)

        # Filter valid SMILES and canonicalize
        """gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)
        if MW_filter:
            gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI[0], gen_conv_SMI, token_probs)
        if MF_filter:
            gen_conv_SMI, token_probs = filter_for_MF_2(trg_conv_SMI[0], gen_conv_SMI, token_probs)
        """
        gen_conv_SMI_list.extend(gen_conv_SMI)
        trg_conv_SMI_list.extend(trg_conv_SMI)
        prob_list.extend(token_probs)
        src_HSQC_list.extend(src_HSQC)
        src_COSY_list.extend(src_COSY)
        
    tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
    results_dict = {
        'gen_conv_SMI_list': gen_conv_SMI_list,
        'trg_conv_SMI_list': trg_conv_SMI_list,
        'tanimoto_sim': tanimoto_list_all,
        'tanimoto_mean': tanimoto_mean,
        'tanimoto_std_dev': tanimoto_std_dev,
        'failed': failed,
        'prob_list': prob_list,
        "src_HSQC_list": src_HSQC_list,
        "src_COSY_list": src_COSY_list,
        }
    
    return config, results_dict
