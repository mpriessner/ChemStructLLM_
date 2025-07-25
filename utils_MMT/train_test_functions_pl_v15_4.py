# Standard libraries
import sys
import json
import random

# Third-party libraries
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Local module imports
#sys.path.append('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer')
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Build the path to the MultiModalTransformer directory relative to the script's location
base_path = os.path.abspath(os.path.join(script_dir, '../'))

# Add the MultiModalTransformer directory to sys.path
if base_path not in sys.path:
    sys.path.append(base_path)

import utils_MMT.helper_functions_pl_v15_4 as hf
# import utils.accuracy_metrics as am
# import utils.eval_functions as ef
import utils_MMT.sgnn_code_pl_v15_4 as sc
import utils_MMT.similarity_functions_v15_4 as sf
import torch.autograd.profiler as profiler
import time

# def calculate_total_loss(gen_output, 
#                          trg_enc_SMI, 
#                         gen_FP, trg_FP, 
#                         gen_conv_SMI, 
#                         trg_conv_SMI, 
#                         trg_MW, 
#                         FP_loss_fn, 
#                         SMI_loss_fn,
#                         MW_loss_fn, 
#                         sgnn_means_stds, 
#                         src_HSQC, 
#                         config, 
#                         batch):
#     """
#     Calculate the total loss for the model.

#     Args:
#         gen_output: Generated output tensor.
#         trg_enc_SMI: Target encoded SMILES.
#         gen_FP: Generated fingerprint.
#         trg_FP: Target fingerprint.
#         gen_conv_SMI: Generated converted SMILES.
#         trg_conv_SMI: Target converted SMILES.
#         trg_MW: Target molecular weight.
#         FP_loss_fn: Fingerprint loss function.
#         SMI_loss_fn: SMILES loss function.
#         MW_loss_fn: Molecular weight loss function.
#         sgnn_means_stds: Mean and standard deviations for SGNN.
#         src_HSQC: Source HSQC data.
#         config: Configuration parameters.
#         batch: Current batch number.

#     Returns:
#         A list of losses and additional metrics.
#     """

#     # Initialize default values and calculate losses
#     sgnn_avg_sim_error_HSQC = None
#     sgnn_avg_sim_error_COSY = None
#     validity_term = 0.0
#     count_reward = 0.0
#     tanimoto_mean = 0.0
#     fingerprint_loss = 0.0
#     mol_weight_loss = torch.tensor([0.0])
#     smi_loss = 0.0
#     fp_loss = 0.0
#     weight_loss = 0.0
#     sgnn_loss = 0.0
#     tanimoto_loss = 0.0
#     valitity_loss = 0.0
#     mol_weight_loss_num = 0.0
#     df_succ_smis = pd.DataFrame()

#     # Update weights based on batch and configuration
#     if batch % config.batch_frequency == 0 and config.change_loss_weights == True:

#         # because the loss is because of the normalization quite small
#         config.weight_MW = min(100, config.weight_MW + (config.increment*100)) 
#         # becasue the loss if quite small because of the averaging of the normalized error
#         config.weight_sgnn = min(10, config.weight_sgnn + config.increment*10) 
#         #config.weight_tanimoto  = min(1, config.weight_tanimoto + config.increment)
#         #config.weight_validity = min(1, config.weight_validity + config.increment)
#         print(f"batch: {batch} | weight_mol_weight: {config.weight_MW} | weight_sgnn: {config.weight_sgnn} | weight_tanimoto: {config.weight_tanimoto} | weight_validity: {config.weight_validity}")
    
#     # If not including weight, set weight factor to zero
#     #if not config.include_weight:
#     #    config.weight_mol_weight = 0.0

#     # Calculate cross-entropy loss
#     output_vector = gen_output.reshape(-1, gen_output.shape[2])
#     target = trg_enc_SMI[1:, :].reshape(-1)
#     smiles_loss = SMI_loss_fn(output_vector, target)
#     #import IPython; IPython.embed();


#     # Perform additional calculations if generating sequence
#     if config.gen_SMI_sequence:
#         try:
#             validity_term, count_reward = create_metric(gen_conv_SMI, trg_conv_SMI)
#             fingerprint_loss = FP_loss_fn(gen_FP, trg_FP)

#             tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = hf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)
#             if len(gen_mol_weights_sel) > 0:

#                 gen_mol_weights_sel = torch.tensor(gen_mol_weights_sel, device=config.device)
#                 trg_mol_weights_sel = torch.tensor(trg_mol_weights_sel, device=config.device)
#                 ### min_max normalization
#                 min_weight = torch.min(torch.min(gen_mol_weights_sel), torch.tensor(config.train_weight_min))
#                 max_weight = torch.max(torch.max(gen_mol_weights_sel), torch.tensor(config.train_weight_max))

#                 gen_mol_weights_sel = (gen_mol_weights_sel - min_weight) / (max_weight - min_weight)
#                 trg_mol_weights_sel = (trg_mol_weights_sel - min_weight) / (max_weight - min_weight)

#                 mol_weight_loss = MW_loss_fn(gen_mol_weights_sel, trg_mol_weights_sel)
#                 mol_weight_loss_num = mol_weight_loss.item()
#                 start = time.time()
#             else:
#                 mol_weight_loss_num = 0.0
#         except:
#             print("there is an error in the gen_sequence part of the calculate loss function")
#             # print(gen_smis)
#             #import IPython; IPython.embed();

#         if config.sgnn_feedback and len(df_succ_smis) >0:
#             #import IPython; IPython.embed();
#             try:
#                 tensor_HSQC = []
#                 for src in src_HSQC:
#                     # Find the rows where both elements are not zero
#                     mask = (src != 0).all(dim=1)
#                     # Use boolean indexing to get the rows
#                     filtered_tensor = src[mask]
#                     if list(filtered_tensor) != []:
#                         list_tensors = [filtered_tensor[:,0]*10, filtered_tensor[:,1]*200]
#                         reshaped_tensors = [tensor.unsqueeze(1) for tensor in list_tensors]
#                         # Concatenate the tensors along the second dimension (columns)
#                         combined_tensor = torch.cat(reshaped_tensors, dim=1)
#                         tensor_HSQC.append(combined_tensor)
#                     else:
#                         tensor_HSQC.append([])
                        
#                 sgnn_avg_sim_error_HSQC, sim_error_list = run_sgnn_sim_calculations_if_possible_2D(df_succ_smis, tensor_HSQC, sgnn_means_stds, config, "HSQC")
#                 sgnn_avg_sim_error_COSY, sim_error_list = run_sgnn_sim_calculations_if_possible_2D(df_succ_smis, tensor_HSQC, sgnn_means_stds, config, "COSY")                                    
                
#             except:
#                 print("there is an error in the run_sgnn_sim_calculations_if_possible part of the calculate loss function")
#                 #import IPython; IPython.embed();
#                 #print(df_succ_smis)

#     # Calculate the total loss and return
#     smi_loss = smiles_loss * config.weight_SMI 
#     fp_loss = fingerprint_loss * config.weight_FP
#     weight_loss = mol_weight_loss_num * config.weight_MW 
#     ### because it could happen that sgnn_avg_sim_error= None because of division by 0 if non of the generated smiles is valid
#     sgnn_loss_HSQC = 0 if sgnn_avg_sim_error_HSQC is None else sgnn_avg_sim_error_HSQC * config.weight_sgnn  
#     sgnn_loss_COSY = 0 if sgnn_avg_sim_error_COSY is None else sgnn_avg_sim_error_COSY * config.weight_sgnn  
#     sgnn_loss = (sgnn_loss_HSQC + sgnn_loss_COSY)/2
#     ###################### maybe add also COSY error ##############################

#     #sgnn_loss = sgnn_avg_sim_error * config.weight_sgnn
#     tanimoto_loss = (1-tanimoto_mean) * config.weight_tanimoto
#     valitity_loss = (1-validity_term) * config.weight_validity
#     total_loss = smi_loss + weight_loss + sgnn_loss #+ fp_loss   + tanimoto_loss + valitity_loss
#     losses_list = [total_loss, smi_loss, fp_loss, weight_loss, sgnn_loss, tanimoto_loss, valitity_loss]
#     #import IPython; IPython.embed();

#     # Print debugging info every 100 batches
#     if batch % 100 == 0:
#         print(f"Total Loss: {total_loss} | smi_loss: {smi_loss} | weight_loss: {weight_loss} | sgnn_loss: {sgnn_loss} | tanimoto_loss: {tanimoto_loss} | valitity_loss: {valitity_loss}")
#         print(f"Cuda: {target.device}")
#         if config.gen_SMI_sequence and gen_conv_SMI != None:
#             print(f"gen_smis[0]: {gen_conv_SMI[0:3]}")
#             print(f"trg_smis[0]: {trg_conv_SMI[0:3]}")

#     return losses_list, mol_weight_loss_num, validity_term, count_reward, tanimoto_mean, sgnn_avg_sim_error_HSQC


    

def run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, sgnn_means_stds, config):
    #import IPython; IPython.embed();
    try:
        batch_data, failed_ids = sc.main_execute(df_succ_smis, sgnn_means_stds, config.ML_dump_folder, int(config.batch_size / config.gpu_num))
    except Exception as e:
        # Handle exceptions from batch data execution
        return None, None, None

    ### run the single generations 
    df_failed = df_succ_smis[df_succ_smis['sample-id'].isin(failed_ids)]   
    try:
         batch_data_add_1, failed_ids = sc.main_execute(df_failed, sgnn_means_stds, config.ML_dump_folder, 10)
    except Exception as e:
        # Handle exceptions from batch data execution
        return None, None, None

    ### run the single generations 
    df_failed = df_succ_smis[df_succ_smis['sample-id'].isin(failed_ids)]   
    try:
         batch_data_add_2, failed_ids = sc.main_execute(df_failed, sgnn_means_stds, config.ML_dump_folder, 1)
    except Exception as e:
        # Handle exceptions from batch data execution
        return None, None, None

    batch_data = pd.concat([batch_data, batch_data_add_1, batch_data_add_2], ignore_index=True)
    ### calculate error of spectra with HungDist method
    matching_technique = config.matching
    padding_technique = config.padding
    mode = matching_technique+"_"+padding_technique
    
    mode_dict = {"MinSum_Zero":0, "EucDis_Zero":1, "HungDist_Zero":2, "MinSum_Trunc":3,   "EucDis_Trunc":4, "HungDist_Trunc":5, "MinSum_NN":6, "EucDis_NN":7, "HungDist_NN":8}

    mode_index = mode_dict[mode]
    sim_error_list = []
    count = 0
    
    for sample_id, sdf_path in zip(batch_data["sample-id"],batch_data["sdf_path"]):
        idx = sample_id.split("_")[0]
        
        #try: ### Because some of the data will be blanked out from the dataloader if selected which would cause an error when this index is selected.
        try: 
            #import IPython; IPython.embed();
            sgnn_sim_df = sf.load_HSQC_dataframe_from_file(sdf_path)
        # Convert tensor to numpy array
            numpy_HSQC = tensor_HSQC[int(idx)].cpu().numpy()
            # Convert numpy array to DataFrame
            if numpy_HSQC.shape[-1]==3:
                pd_df_HSQC = pd.DataFrame(numpy_HSQC, columns=['F2 (ppm)', 'F1 (ppm)', "direction"])
            elif numpy_HSQC.shape[-1]==2:
                pd_df_HSQC = pd.DataFrame(numpy_HSQC, columns=['F2 (ppm)', 'F1 (ppm)'])

            sim_error, _ = sf.get_similarity_comparison_variations(pd_df_HSQC, sgnn_sim_df, mode, idx, similarity_type="euclidean", error="avg", display_img=False)
            sim_error_list.append(sim_error[mode_index])
            count +=1
        except:
            #print("tensor_HSQC blanked out")
            #print(tensor_HSQC[int(idx)])
            #print(sdf_path)
            #import IPython; IPython.embed();
            pass
    if count != 0:
        #print(len(batch_data))
        #print(len(sim_error_list))
        avg_sim_error = sum(sim_error_list)/count
        sim_error_list = [float(x) for x in sim_error_list]

        return avg_sim_error, sim_error_list
    else:
        return None, None

def run_sgnn_sim_calculations_if_possible_return_spectra(df_succ_smis, tensor_HSQC, tensor_COSY, sgnn_means_stds, config):
    """
    Perform sgnn similarity calculations if possible.

    Args:
        df_succ_smis (pd.DataFrame): DataFrame of successful SMILES.
        tensor_HSQC (Tensor): Tensor of HSQC data.
        sgnn_means_stds (dict): Dictionary of means and standard deviations for sgnn.
        config (object): Configuration object containing various settings.

    Returns:
        tuple: Average similarity error and a list of similarity errors, or None if not applicable.
    """

    try:
        print("run_sgnn_sim_calculations_if_possible_return_spectra")
        # import IPython; IPython.embed();
        parent_dir = os.path.dirname(config.ML_dump_folder)  # This gives you the 'exp_1' directory
        # Set the new dump folder path
        dump_folder = os.path.join(parent_dir, f"SGNN_gen_folder_{config.ran_num}", f"sdf_{config.ran_num}")

        batch_data, failed_ids = sc.main_execute(df_succ_smis, sgnn_means_stds, dump_folder, int(config.batch_size / config.gpu_num))
    except Exception as e:
        # Handle exceptions from batch data execution
        return None, None, None, None, None

    ### run the single generations 
    df_failed = df_succ_smis[df_succ_smis['sample-id'].isin(failed_ids)]   
    try:
         batch_data_add_1, failed_ids = sc.main_execute(df_failed, sgnn_means_stds, dump_folder, 10)
    except Exception as e:
        # Handle exceptions from batch data execution
        return None, None, None, None, None

    ### run the single generations 
    df_failed = df_succ_smis[df_succ_smis['sample-id'].isin(failed_ids)]   
    try:
         batch_data_add_2, failed_ids = sc.main_execute(df_failed, sgnn_means_stds, dump_folder, 1)
    except Exception as e:
        # Handle exceptions from batch data execution
        return None, None, None, None, None

    batch_data = pd.concat([batch_data, batch_data_add_1, batch_data_add_2], ignore_index=True)
    mode = f"{config.matching}_{config.padding}"

    mode_dict = {
        "MinSum_Zero": 0, "EucDis_Zero": 1, "HungDist_Zero": 2,
        "MinSum_Trunc": 3, "EucDis_Trunc": 4, "HungDist_Trunc": 5,
        "MinSum_NN": 6, "EucDis_NN": 7, "HungDist_NN": 8
    }

    mode_index = mode_dict.get(mode, -1)
    sim_error_list_HSQC = []
    sim_error_list_COSY = []
    count = 0

    for sample_id, sdf_path in zip(batch_data["sample-id"],batch_data["sdf_path"]):
        idx = sample_id.split("_")[0]

        try:
            sgnn_sim_df = sf.load_HSQC_dataframe_from_file(sdf_path)
            numpy_HSQC = tensor_HSQC[int(idx)].cpu().numpy()
            columns = ['F2 (ppm)', 'F1 (ppm)'] + (["direction"] if numpy_HSQC.shape[-1] == 3 else [])
            pd_df_HSQC = pd.DataFrame(numpy_HSQC, columns=columns)

            sim_error, _ = sf.get_similarity_comparison_variations(pd_df_HSQC, sgnn_sim_df, mode, idx, similarity_type="euclidean", error="avg", display_img=False)
            sim_error_list_HSQC.append(sim_error[mode_index])
            count += 1
        except Exception as e:
            # Handle exceptions from similarity calculations
            sim_error_list_HSQC.append(9)  # arbitrary high number
            continue
            #import IPython; IPython.embed();


    for sample_id, sdf_path in zip(batch_data["sample-id"],batch_data["sdf_path"]):
        idx = sample_id.split("_")[0]

        try:
            sgnn_sim_df = sf.load_COSY_dataframe_from_file(sdf_path)
            numpy_COSY = tensor_COSY[int(idx)].cpu().numpy()
            columns = ['F2 (ppm)', 'F1 (ppm)'] + (["direction"] if numpy_COSY.shape[-1] == 3 else [])
            pd_df_COSY = pd.DataFrame(numpy_COSY, columns=columns)

            sim_error_CSOY, _ = sf.get_similarity_comparison_variations(pd_df_COSY, sgnn_sim_df, mode, idx, similarity_type="euclidean", error="avg", display_img=False)
            sim_error_list_COSY.append(sim_error_CSOY[mode_index])
            count += 1
        except Exception as e:
            # Handle exceptions from similarity calculations
            sim_error_list_COSY.append(9)  # arbitrary high number
            continue
            #import IPython; IPython.embed();            

    if count == 0:
        return None, None, None, None, None
    avg_sim_error_HSQC = sum(sim_error_list_HSQC)/count
    sim_error_list_HSQC = [float(x) for x in sim_error_list_HSQC]
    
    avg_sim_error_COSY = sum(sim_error_list_COSY)/count
    sim_error_list_COSY = [float(x) for x in sim_error_list_COSY]

    return avg_sim_error_HSQC, avg_sim_error_COSY, sim_error_list_HSQC, sim_error_list_COSY, batch_data

'''
def create_metric(gen_smis, trg_smis):
    """
    Calculate metrics for the generated SMILES strings.

    :param gen_smis: List of generated SMILES strings.
    :param trg_smis: List of target SMILES strings.
    :return: Tuple containing validity term and count-based reward.
    """

    # Guard clause for empty input lists
    if not gen_smis or not trg_smis:
        return 0.0, 0.0

    # Calculate the validity score for the generated SMILES
    validity_term = hf.get_validity_term(gen_smis)

    # Calculate the reward based on the correct symbols in generated SMILES
    count_reward = hf.count_based_reward(gen_smis, trg_smis)

    return validity_term, count_reward'''

    


# def tensor_to_smiles(tensor, itos):
#     """
#     Converts a tensor of token IDs to a SMILES string or list of SMILES strings.

#     :param tensor: Tensor containing token IDs.
#     :param itos: Dictionary mapping token IDs to SMILES tokens.
#     :return: Single SMILES string or list of SMILES strings.
#     """
#     if len(tensor.shape)>1:
#         sequence_length, batch_size = tensor.shape
#         sequences = []

#         for i in range(batch_size):
#             sequence = []
#             for j in range(0, sequence_length): 
#                 token = itos[str(int(tensor[j, i].item()))]
#                 if token == "<EOS>":
#                     break
#                 sequence.append(token)
#             sequences.append("".join(sequence))
        
#         return sequences
#     else: 
#         sequence = []
#         for j in range(0, len(tensor)): 
#             token = itos[str(int(tensor[j].item()))]        
#             if token == "<EOS>":
#                 break
#             sequence.append(token)
#         smi = "".join(sequence)
#         return smi


# def tensor_to_smiles_and_prob(tensor, token_prob, itos):
#     if len(tensor.shape) > 1:
#         # Convert tensor to numpy for efficient access
#         tensor_np = tensor.cpu().numpy()
#         sequences = []
#         token_prob_cut = []

#         for i in range(tensor.shape[1]):  # Iterate over batch_size
#             # Find the index of the first occurrence of <EOS> token
#             eos_idx = next((idx for idx, val in enumerate(tensor_np[:, i]) if itos[str(val)] == "<EOS>"), tensor.shape[0])
            
#             # Build sequence using list comprehension
#             sequence = [itos[str(val)] for val in tensor_np[:eos_idx, i]]
#             sequences.append("".join(sequence))

#             # Slice the probability tensor
#             token_prob_cut.append(token_prob[i, :eos_idx])

#         return sequences, token_prob_cut

#     else:
#         # Handle the case for a single sequence
#         tensor_np = tensor.cpu().numpy()
#         eos_idx = next((idx for idx, val in enumerate(tensor_np) if itos[str(val)] == "<EOS>"), len(tensor))
        
#         sequence = [itos[str(val)] for val in tensor_np[:eos_idx]]
#         smi = "".join(sequence)
#         token_prob_list = list(token_prob[:eos_idx])

#         return smi, torch.stack(token_prob_list)
    
    
'''
def tensor_to_smiles_and_prob(tensor, token_prob, itos):
    """
    Converts a tensor of token IDs to a SMILES string or list of SMILES strings and their corresponding probabilities.

    :param tensor: Tensor containing token IDs.
    :param token_prob: Tensor containing token probabilities.
    :param itos: Dictionary mapping token IDs to SMILES tokens.
    :return: Tuple of SMILES strings and their probabilities.
    """
    token_prob_cut = []
    if len(tensor.shape)>1:
        sequence_length, batch_size = tensor.shape
        sequences = []
        for i in range(batch_size):
            sequence = []
            for j in range(0, sequence_length): 
                token = itos[str(int(tensor[j, i].item()))]
                if token == "<EOS>":
                    break
                sequence.append(token)
            sequences.append("".join(sequence))
            token_prob_cut.append(token_prob[i,:len(sequence)])

        return sequences, token_prob_cut
    else: 
        sequence = []
        for j in range(0, len(tensor)): 
            token = itos[str(int(tensor[j].item()))]     
            if token == "<EOS>":
                break
            sequence.append(token)
            token_prob_cut.append(token_prob[j])
        smi = "".join(sequence)
        token_prob_list = list(token_prob[:len(sequence)])
        return smi, torch.stack(token_prob_list)        
        #return smi, torch.stack(token_prob[:len(sequence)])        

'''