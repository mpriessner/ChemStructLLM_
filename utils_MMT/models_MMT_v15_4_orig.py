# before modifying _embed_spectrum_13C for single sample processing

import argparse
import math
import os
import random
import sys
import time
import datetime
import json

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_MMT.helper_functions_pl_v15_4 as hf
from utils_MMT.sgnn_code_pl_v15_4 import load_std_mean
import utils_MMT.sgnn_code_pl_v15_4 as sc

#from utils.models_CLIP_v14_1 import CLIPModel

from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.autograd.profiler as profiler



# Adding project-specific path to sys.path
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

def make_pl(config):
    
    ### Load dictionaries:
    # Load from file
    with open(config.itos_path, 'r') as f:
        itos = json.load(f)

    with open(config.stoi_path, 'r') as f:
        stoi = json.load(f)
        
    with open(config.itos_MF_path, 'r') as f:
        itos_MF = json.load(f)

    with open(config.stoi_MF_path, 'r') as f:
        stoi_MF = json.load(f)        

    # Load Model
    #model = MultimodalTransformer(config, src_pad_idx=0)
        
    # Define a loss function    
    SMI_loss_fn = torch.nn.CrossEntropyLoss().to(config.device)         
    #smiles_loss_fn = nn.NLLLoss(reduction="none", ignore_index=0)
    MF_loss_fn = torch.nn.CrossEntropyLoss().to(config.device)       #not used yet          
    MW_loss_fn = torch.nn.MSELoss().to(config.device)    
    FP_loss_fn = torch.nn.BCEWithLogitsLoss().to(config.device)
    

    #Load SGNN mean values
    graph_representation = "sparsified"
    target = "13C"
    #train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H = None, None, None, None
    train_y_mean_C, train_y_std_C = load_std_mean(target,graph_representation)
    target = "1H"
    train_y_mean_H, train_y_std_H = load_std_mean(target,graph_representation)
    sgnn_means_stds = (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)
    
    return SMI_loss_fn, MF_loss_fn, MW_loss_fn, FP_loss_fn, itos, stoi, itos_MF, stoi_MF, sgnn_means_stds


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

    return validity_term, count_reward




def run_sgnn_sim_calculations_if_possible_2D(df_succ_smis, tensor_2D, sgnn_means_stds, config, NMR_technique):
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
        
        if NMR_technique == "HSQC":
            #try: ### Because some of the data will be blanked out from the dataloader if selected which would cause an error when this index is selected.
            try: 
                sgnn_sim_df = sf.load_HSQC_dataframe_from_file(sdf_path)
            # Convert tensor to numpy array
                numpy_HSQC = tensor_2D[int(idx)].cpu().numpy()
                # Convert numpy array to DataFrame
                if numpy_HSQC.shape[-1]==3:
                    pd_df_HSQC = pd.DataFrame(numpy_HSQC, columns=['F2 (ppm)', 'F1 (ppm)', "direction"])
                elif numpy_HSQC.shape[-1]==2:
                    pd_df_HSQC = pd.DataFrame(numpy_HSQC, columns=['F2 (ppm)', 'F1 (ppm)'])

                sim_error, _ = sf.get_similarity_comparison_variations(pd_df_HSQC, sgnn_sim_df, mode, idx, similarity_type="euclidean", error="avg", display_img=False)
                sim_error_list.append(sim_error[mode_index])
                count +=1
            except:
                pass

        elif NMR_technique == "COSY":
            #try: ### Because some of the data will be blanked out from the dataloader if selected which would cause an error when this index is selected.
            try: 
                sgnn_sim_df = sf.load_COSY_dataframe_from_file(sdf_path)
            # Convert tensor to numpy array
                numpy_COSY = tensor_2D[int(idx)].cpu().numpy()
                # Convert numpy array to DataFrame
                if numpy_COSY.shape[-1]==3:
                    pd_df_COSY = pd.DataFrame(numpy_COSY, columns=['F2 (ppm)', 'F1 (ppm)', "direction"])
                elif numpy_COSY.shape[-1]==2:
                    pd_df_COSY = pd.DataFrame(numpy_COSY, columns=['F2 (ppm)', 'F1 (ppm)'])

                sim_error, _ = sf.get_similarity_comparison_variations(pd_df_COSY, sgnn_sim_df, mode, idx, similarity_type="euclidean", error="avg", display_img=False)
                sim_error_list.append(sim_error[mode_index])
                count +=1
            except:
                pass    

    if count != 0:
        #print(len(batch_data))
        #print(len(sim_error_list))
        avg_sim_error = sum(sim_error_list)/count
        sim_error_list = [float(x) for x in sim_error_list]

        return avg_sim_error, sim_error_list
    else:
        return None, None
    

def calculate_total_loss(gen_output, 
                         trg_enc_SMI, 
                        gen_FP, trg_FP, 
                        gen_conv_SMI, 
                        trg_conv_SMI, 
                        trg_MW, 
                        FP_loss_fn, 
                        SMI_loss_fn,
                        MW_loss_fn, 
                        sgnn_means_stds, 
                        src_HSQC, 
                        config, 
                        batch):
    """
    Calculate the total loss for the model.

    Args:
        gen_output: Generated output tensor.
        trg_enc_SMI: Target encoded SMILES.
        gen_FP: Generated fingerprint.
        trg_FP: Target fingerprint.
        gen_conv_SMI: Generated converted SMILES.
        trg_conv_SMI: Target converted SMILES.
        trg_MW: Target molecular weight.
        FP_loss_fn: Fingerprint loss function.
        SMI_loss_fn: SMILES loss function.
        MW_loss_fn: Molecular weight loss function.
        sgnn_means_stds: Mean and standard deviations for SGNN.
        src_HSQC: Source HSQC data.
        config: Configuration parameters.
        batch: Current batch number.

    Returns:
        A list of losses and additional metrics.
    """

    # Initialize default values and calculate losses
    sgnn_avg_sim_error_HSQC = None
    sgnn_avg_sim_error_COSY = None
    validity_term = 0.0
    count_reward = 0.0
    tanimoto_mean = 0.0
    fingerprint_loss = 0.0
    mol_weight_loss = torch.tensor([0.0])
    smi_loss = 0.0
    fp_loss = 0.0
    weight_loss = 0.0
    sgnn_loss = 0.0
    tanimoto_loss = 0.0
    valitity_loss = 0.0
    mol_weight_loss_num = 0.0
    df_succ_smis = pd.DataFrame()

    # Update weights based on batch and configuration
    if batch % config.batch_frequency == 0 and config.change_loss_weights == True:

        # because the loss is because of the normalization quite small
        config.weight_MW = min(100, config.weight_MW + (config.increment*100)) 
        # becasue the loss if quite small because of the averaging of the normalized error
        config.weight_sgnn = min(10, config.weight_sgnn + config.increment*10) 
        #config.weight_tanimoto  = min(1, config.weight_tanimoto + config.increment)
        #config.weight_validity = min(1, config.weight_validity + config.increment)
        print(f"batch: {batch} | weight_mol_weight: {config.weight_MW} | weight_sgnn: {config.weight_sgnn} | weight_tanimoto: {config.weight_tanimoto} | weight_validity: {config.weight_validity}")
    
    # If not including weight, set weight factor to zero
    #if not config.include_weight:
    #    config.weight_mol_weight = 0.0

    # Calculate cross-entropy loss
    output_vector = gen_output.reshape(-1, gen_output.shape[2])
    target = trg_enc_SMI[1:, :].reshape(-1)
    smiles_loss = SMI_loss_fn(output_vector, target)
    #import IPython; IPython.embed();


    # Perform additional calculations if generating sequence
    if config.gen_SMI_sequence:
        try:

            validity_term, count_reward = create_metric(gen_conv_SMI, trg_conv_SMI)
            fingerprint_loss = FP_loss_fn(gen_FP, trg_FP)

            tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = hf.calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW)
            if len(gen_mol_weights_sel) > 0:

                gen_mol_weights_sel = torch.tensor(gen_mol_weights_sel, device=config.device)
                trg_mol_weights_sel = torch.tensor(trg_mol_weights_sel, device=config.device)
                ### min_max normalization
                min_weight = torch.min(torch.min(gen_mol_weights_sel), torch.tensor(config.train_weight_min))
                max_weight = torch.max(torch.max(gen_mol_weights_sel), torch.tensor(config.train_weight_max))

                gen_mol_weights_sel = (gen_mol_weights_sel - min_weight) / (max_weight - min_weight)
                trg_mol_weights_sel = (trg_mol_weights_sel - min_weight) / (max_weight - min_weight)

                mol_weight_loss = MW_loss_fn(gen_mol_weights_sel, trg_mol_weights_sel)
                mol_weight_loss_num = mol_weight_loss.item()
                start = time.time()
            else:
                mol_weight_loss_num = 0.0
        except:
            print("there is an error in the gen_sequence part of the calculate loss function")
            # print(gen_smis)
            # import IPython; IPython.embed();

        if config.sgnn_feedback and len(df_succ_smis) >0:
            #import IPython; IPython.embed();
            try:
                tensor_HSQC = []
                for src in src_HSQC:
                    # Find the rows where both elements are not zero
                    mask = (src != 0).all(dim=1)
                    # Use boolean indexing to get the rows
                    filtered_tensor = src[mask]
                    if list(filtered_tensor) != []:
                        list_tensors = [filtered_tensor[:,0]*10, filtered_tensor[:,1]*200]
                        reshaped_tensors = [tensor.unsqueeze(1) for tensor in list_tensors]
                        # Concatenate the tensors along the second dimension (columns)
                        combined_tensor = torch.cat(reshaped_tensors, dim=1)
                        tensor_HSQC.append(combined_tensor)
                    else:
                        tensor_HSQC.append([])
                        
                sgnn_avg_sim_error_HSQC, sim_error_list = run_sgnn_sim_calculations_if_possible_2D(df_succ_smis, tensor_HSQC, sgnn_means_stds, config, "HSQC")
                sgnn_avg_sim_error_COSY, sim_error_list = run_sgnn_sim_calculations_if_possible_2D(df_succ_smis, tensor_HSQC, sgnn_means_stds, config, "COSY")                                    
                
            except:
                print("there is an error in the run_sgnn_sim_calculations_if_possible part of the calculate loss function")
                #import IPython; IPython.embed();
                #print(df_succ_smis)

    # Calculate the total loss and return
    smi_loss = smiles_loss * config.weight_SMI 
    fp_loss = fingerprint_loss * config.weight_FP
    weight_loss = mol_weight_loss_num * config.weight_MW 
    ### because it could happen that sgnn_avg_sim_error= None because of division by 0 if non of the generated smiles is valid
    sgnn_loss_HSQC = 0 if sgnn_avg_sim_error_HSQC is None else sgnn_avg_sim_error_HSQC * config.weight_sgnn  
    sgnn_loss_COSY = 0 if sgnn_avg_sim_error_COSY is None else sgnn_avg_sim_error_COSY * config.weight_sgnn  
    sgnn_loss = (sgnn_loss_HSQC + sgnn_loss_COSY)/2
    ###################### maybe add also COSY error ##############################

    #sgnn_loss = sgnn_avg_sim_error * config.weight_sgnn
    tanimoto_loss = (1-tanimoto_mean) * config.weight_tanimoto
    valitity_loss = (1-validity_term) * config.weight_validity
    total_loss = smi_loss + weight_loss + sgnn_loss #+ fp_loss   + tanimoto_loss + valitity_loss
    losses_list = [total_loss, smi_loss, fp_loss, weight_loss, sgnn_loss, tanimoto_loss, valitity_loss]
    #import IPython; IPython.embed();

    # Print debugging info every 100 batches
    if batch % 100 == 0:
        print(f"Total Loss: {total_loss} | smi_loss: {smi_loss} | weight_loss: {weight_loss} | sgnn_loss: {sgnn_loss} | tanimoto_loss: {tanimoto_loss} | valitity_loss: {valitity_loss}")
        print(f"Cuda: {target.device}")
        if config.gen_SMI_sequence and gen_conv_SMI != None:
            print(f"gen_smis[0]: {gen_conv_SMI[0:3]}")
            print(f"trg_smis[0]: {trg_conv_SMI[0:3]}")

    return losses_list, mol_weight_loss_num, validity_term, count_reward, tanimoto_mean, sgnn_avg_sim_error_HSQC


class PointEmbedding_1H(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointEmbedding_1H, self).__init__()
        self.fc_H = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc_H(x)
        return x    
    
class PointEmbedding_13C(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointEmbedding_13C, self).__init__()
        self.fc_C = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc_C(x)
        return x
    
class PointEmbedding_HSQC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointEmbedding_HSQC, self).__init__()
        self.fc_HSQC = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc_HSQC(x)
        return x

class PointEmbedding_COSY(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointEmbedding_COSY, self).__init__()
        self.fc_COSY = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc_COSY(x)
        return x

class SpectrumEmbedding_1H(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpectrumEmbedding_1H, self).__init__()
        self.point_embedding_layer_1H = PointEmbedding_1H(input_dim, output_dim)

    def forward(self, padded_spectra):
        embedded_spectra = self.point_embedding_layer_1H(padded_spectra)
        embedded_spectra = F.relu(embedded_spectra)          
        return embedded_spectra

class SpectrumEmbedding_13C(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpectrumEmbedding_13C, self).__init__()
        self.point_embedding_layer_13C = PointEmbedding_13C(input_dim, output_dim)

    def forward(self, padded_spectra):
        padded_spectra1 = padded_spectra.unsqueeze(-1)
        embedded_spectra = self.point_embedding_layer_13C(padded_spectra1)
        embedded_spectra = F.relu(embedded_spectra)      

        return embedded_spectra

class SpectrumEmbedding_HSQC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpectrumEmbedding_HSQC, self).__init__()
        self.point_embedding_layer_HSQC = PointEmbedding_HSQC(input_dim, output_dim)

    def forward(self, padded_spectra):
        embedded_spectra = self.point_embedding_layer_HSQC(padded_spectra)
        embedded_spectra = F.relu(embedded_spectra)

        return embedded_spectra
    
class SpectrumEmbedding_COSY(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpectrumEmbedding_COSY, self).__init__()
        self.point_embedding_layer_COSY = PointEmbedding_COSY(input_dim, output_dim)

    def forward(self, padded_spectra):
        embedded_spectra = self.point_embedding_layer_COSY(padded_spectra)
        embedded_spectra = F.relu(embedded_spectra)     
        return embedded_spectra  
    
class SpectrumEmbedding_IR(nn.Module):
    def __init__(self, num_bins, embedding_dim):
        super(SpectrumEmbedding_IR, self).__init__()
        self.linear_spec_embedding_IR = nn.Linear(num_bins, embedding_dim)
    
    def forward(self, x):
        embedded_spectrum = self.linear_spec_embedding_IR(x.float())
        return embedded_spectrum

class MolecularFormulaEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MolecularFormulaEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        #self.embedding_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.embedding(x) 
        return x  

class MolecularSmilesEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MolecularSmilesEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        #self.embedding_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.embedding(x) 
        return x  
    
class Embedding_MW(nn.Module):
    def __init__(self, num_bins, embedding_dim):
        super(Embedding_MW, self).__init__()
        self.linear_spec_embedding_MW = nn.Linear(num_bins, embedding_dim)

    def forward(self, x):
        x = x.float()  # Cast to float if necessary
        embedded_spectrum = self.linear_spec_embedding_MW(x)
        return embedded_spectrum

class MolecularSmilesEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MolecularSmilesEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        #self.embedding_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.embedding(x) 
        return x  

    
class MultimodalTransformer(nn.Module):
    def __init__(self, config, src_pad_idx=0):
        super(MultimodalTransformer, self).__init__()

        self.config = config
        #self.src_pad_idx = src_pad_idx

        # Embedding layers inputs
        self.linear_spec_embedding_1H = SpectrumEmbedding_1H(config.input_dim_1H, config.hidden_size)
        self.linear_spec_embedding_13C = SpectrumEmbedding_13C(config.input_dim_13C, config.hidden_size)
        self.linear_spec_embedding_HSQC = SpectrumEmbedding_HSQC(config.input_dim_HSQC, config.hidden_size)
        self.linear_spec_embedding_COSY = SpectrumEmbedding_COSY(config.input_dim_COSY, config.hidden_size)
        self.linear_spec_embedding_IR = SpectrumEmbedding_IR(config.input_dim_IR, config.hidden_size)
        self.linear_embedding_MF = MolecularFormulaEmbedding(config.MF_vocab_size, config.hidden_size)
        self.linear_embedding_MS = MolecularSmilesEmbedding(config.MS_vocab_size, config.hidden_size)
        self.linear_embedding_MW = Embedding_MW(1, config.hidden_size)

        # Target embedding
        self.embed_trg = nn.Embedding(config.in_size, config.hidden_size)
        self.pe_trg = nn.Embedding(config.max_len, config.hidden_size)

        # Transformer
        # Separate encoders for each data type
        self.encoder_1H = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads),
            num_layers=config.num_encoder_layers)
        
        self.encoder_13C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads),
            num_layers=config.num_encoder_layers)
        
        self.encoder_HSQC = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads),
            num_layers=config.num_encoder_layers)

        self.encoder_COSY = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads),
            num_layers=config.num_encoder_layers)
                        
        self.encoder_IR = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads),
            num_layers=config.num_encoder_layers)
        
        ####################
        self.encoder_cross = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=int(config.num_heads/4)),
            num_layers=config.num_encoder_layers)        
        ####################
        

        
        # Initialize separate decoders
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_heads),
            num_layers=config.num_decoder_layers)  

        self.fp1 = nn.Linear(config.hidden_size, config.fingerprint_size)
        self.dropout2 = nn.Dropout(config.drop_out)
        self.fc_out = nn.Linear(config.hidden_size, config.out_size)
        self.real_data_linear = nn.Linear(config.hidden_size, config.out_size)
        
    
    def _create_embeddings_and_masks_1H(self, embedding_src_1H=None, src_padding_mask_1H=None, 
                                     embedding_src_MF=None, src_padding_mask_MF=None,
                                     embedding_src_MS=None, src_padding_mask_MS=None,
                                     embedding_src_MW=None, src_padding_mask_MW=None
                                     ):

        # Initialize empty lists to collect embeddings and masks
        embeddings = []
        masks = []

        # Fetch embeddings and masks based on the mode
        if "1H" in self.config.training_mode and embedding_src_1H != None:
            embeddings.append(embedding_src_1H)
            masks.append(src_padding_mask_1H)
        if "MF" in self.config.training_mode and embedding_src_MF != None:
            embeddings.append(embedding_src_MF)
            masks.append(src_padding_mask_MF) 
        if "MS" in self.config.training_mode and embedding_src_MS != None:
            embeddings.append(embedding_src_MS)
            masks.append(src_padding_mask_MS)  
        if "MW" in self.config.training_mode and embedding_src_MW != None:
            embeddings.append(embedding_src_MW)
            masks.append(src_padding_mask_MW)

            #print("model")    
            #import IPython; IPython.embed();

        # Concatenate all the collected embeddings and masks along the appropriate dimensions
        if embeddings and masks:
            #import IPython; IPython.embed();            
            embedding_src = torch.cat(embeddings, dim=0)
            src_padding_mask = torch.cat(masks, dim=1)         
            return embedding_src, src_padding_mask
        else:
            print("No valid modes provided. Returning None.")
            return None, None
        

    def _create_embeddings_and_masks_13C(self,embedding_src_13C=None, src_padding_mask_13C=None,
                                     embedding_src_MF=None, src_padding_mask_MF=None,
                                     embedding_src_MS=None, src_padding_mask_MS=None,
                                     embedding_src_MW=None, src_padding_mask_MW=None
                                     ):

        # Initialize empty lists to collect embeddings and masks
        embeddings = []
        masks = []

        # Fetch embeddings and masks based on the mode
        if "13C" in self.config.training_mode and embedding_src_13C != None:
            embeddings.append(embedding_src_13C)
            masks.append(src_padding_mask_13C)
        if "MF" in self.config.training_mode and embedding_src_MF != None:
            embeddings.append(embedding_src_MF)
            masks.append(src_padding_mask_MF) 
        if "MS" in self.config.training_mode and embedding_src_MS != None:
            embeddings.append(embedding_src_MS)
            masks.append(src_padding_mask_MS)  
        if "MW" in self.config.training_mode and embedding_src_MW != None:
            embeddings.append(embedding_src_MW)
            masks.append(src_padding_mask_MW)

            #print("model")    
            #import IPython; IPython.embed();

        # Concatenate all the collected embeddings and masks along the appropriate dimensions
        if embeddings and masks:
            embedding_src = torch.cat(embeddings, dim=0)
            src_padding_mask = torch.cat(masks, dim=1)         
            return embedding_src, src_padding_mask
        else:
            print("No valid modes provided. Returning None.")
            return None, None


    def _create_embeddings_and_masks_HSQC(self, embedding_src_HSQC=None, src_padding_mask_HSQC=None, 
                                     embedding_src_MF=None, src_padding_mask_MF=None,
                                     embedding_src_MS=None, src_padding_mask_MS=None,
                                     embedding_src_MW=None, src_padding_mask_MW=None
                                     ):

        # Initialize empty lists to collect embeddings and masks
        embeddings = []
        masks = []

        # Loop through each mode specified in self.config.training_mode
        # Fetch embeddings and masks based on the mode
        if "HSQC" in self.config.training_mode and embedding_src_HSQC != None:
            embeddings.append(embedding_src_HSQC)
            masks.append(src_padding_mask_HSQC)
        if "MF" in self.config.training_mode and embedding_src_MF != None:
            embeddings.append(embedding_src_MF)
            masks.append(src_padding_mask_MF) 
        if "MS" in self.config.training_mode and embedding_src_MS != None:
            embeddings.append(embedding_src_MS)
            masks.append(src_padding_mask_MS)  
        if "MW" in self.config.training_mode and embedding_src_MW != None:
            embeddings.append(embedding_src_MW)
            masks.append(src_padding_mask_MW)
                
            #print("model")    
            #import IPython; IPython.embed();

        # Concatenate all the collected embeddings and masks along the appropriate dimensions
        if embeddings and masks:
            embedding_src = torch.cat(embeddings, dim=0)
            src_padding_mask = torch.cat(masks, dim=1)         
            return embedding_src, src_padding_mask
        else:
            print("No valid modes provided. Returning None.")
            return None, None


    def _create_embeddings_and_masks_COSY(self, embedding_src_COSY=None, src_padding_mask_COSY=None,
                                     embedding_src_MF=None, src_padding_mask_MF=None,
                                     embedding_src_MS=None, src_padding_mask_MS=None,
                                     embedding_src_MW=None, src_padding_mask_MW=None
                                     ):

        # Initialize empty lists to collect embeddings and masks
        embeddings = []
        masks = []

        # Fetch embeddings and masks based on the mode
        if "COSY" in self.config.training_mode and embedding_src_COSY != None:
            embeddings.append(embedding_src_COSY)
            masks.append(src_padding_mask_COSY)
        if "MF" in self.config.training_mode and embedding_src_MF != None:
            embeddings.append(embedding_src_MF)
            masks.append(src_padding_mask_MF) 
        if "MS" in self.config.training_mode and embedding_src_MS != None:
            embeddings.append(embedding_src_MS)
            masks.append(src_padding_mask_MS)  
        if "MW" in self.config.training_mode and embedding_src_MW != None:
            embeddings.append(embedding_src_MW)
            masks.append(src_padding_mask_MW)
                
            #print("model")    
            #import IPython; IPython.embed();

        # Concatenate all the collected embeddings and masks along the appropriate dimensions
        if embeddings and masks:
            embedding_src = torch.cat(embeddings, dim=0)
            src_padding_mask = torch.cat(masks, dim=1)         
            return embedding_src, src_padding_mask
        else:
            print("No valid modes provided. Returning None.")
            return None, None

    def _create_embeddings_and_masks_IR(self, embedding_src_IR=None, src_padding_mask_IR=None,
                                     embedding_src_MF=None, src_padding_mask_MF=None,
                                     embedding_src_MS=None, src_padding_mask_MS=None,
                                     embedding_src_MW=None, src_padding_mask_MW=None
                                     ):

        # Initialize empty lists to collect embeddings and masks
        embeddings = []
        masks = []

        if "IR" in self.config.training_mode and embedding_src_IR != None:
            embeddings.append(embedding_src_IR)
            masks.append(src_padding_mask_IR)
        if "MF" in self.config.training_mode and embedding_src_MF != None:
            embeddings.append(embedding_src_MF)
            masks.append(src_padding_mask_MF) 
        if "MS" in self.config.training_mode and embedding_src_MS != None:
            embeddings.append(embedding_src_MS)
            masks.append(src_padding_mask_MS)  
        if "MW" in self.config.training_mode and embedding_src_MW != None:
            embeddings.append(embedding_src_MW)
            masks.append(src_padding_mask_MW)
                
            #print("model")    
            #import IPython; IPython.embed();

        # Concatenate all the collected embeddings and masks along the appropriate dimensions
        if embeddings and masks:
            embedding_src = torch.cat(embeddings, dim=0)
            src_padding_mask = torch.cat(masks, dim=1)         
            return embedding_src, src_padding_mask
        else:
            print("No valid modes provided. Returning None.")
            return None, None

    def _embed_spectrum_1H(self, spectrum, mask):
        embedding = self.linear_spec_embedding_1H(spectrum)
        embedding = F.relu(embedding) 
        embedding = embedding.permute(1,0,2)
        padding_mask = mask.to(torch.bool)
        return embedding, padding_mask
    
    def _embed_spectrum_13C(self, spectrum, mask):
        embedding = self.linear_spec_embedding_13C(spectrum)
        embedding = F.relu(embedding)
        embedding = embedding.permute(1,0,2)
        padding_mask = mask.to(torch.bool)
        return embedding, padding_mask
    
    def _embed_spectrum_HSQC(self, spectrum, mask):
        embedding = self.linear_spec_embedding_HSQC(spectrum)
        embedding = F.relu(embedding) 
        embedding = embedding.permute(1,0,2)
        padding_mask = mask.to(torch.bool)
        return embedding, padding_mask

    def _embed_spectrum_COSY(self, spectrum, mask):
        embedding = self.linear_spec_embedding_COSY(spectrum)
        embedding = F.relu(embedding) 
        embedding = embedding.permute(1,0,2)
        padding_mask = mask.to(torch.bool)
        return embedding, padding_mask
    
    def _embed_spectrum_IR(self, spectrum, mask):
        # Apply linear transformation
        embedding = self.linear_spec_embedding_IR(spectrum)
        embedding = F.relu(embedding)
        # Since I look at all the points in the spectrum I make all true
        padding_mask = torch.zeros(len(embedding), dtype=torch.bool, device=self.config.device)
        return embedding, padding_mask

    def _embed_MF(self, tokenized_formula, padding_mask):
        embedding = self.linear_embedding_MF(tokenized_formula)
        embedding = F.relu(embedding)            
        #embedding = embedding.unsqueeze(0)   
        embedding = embedding.permute(1, 0, 2)
        #padding_mask = torch.zeros(len(embedding), dtype=torch.bool, device=self.config.device)
        #padding_mask = padding_mask.squeeze(0)
        return embedding, padding_mask    

    def _embed_MS(self, tokenized_smi, padding_mask):
        embedding = self.linear_embedding_MS(tokenized_smi)
        embedding = F.relu(embedding)            
        #embedding = embedding.unsqueeze(0)   
        embedding = embedding.permute(1, 0, 2)
        #padding_mask = torch.zeros(len(embedding), dtype=torch.bool, device=self.config.device)
        #padding_mask = padding_mask.squeeze(0)
        return embedding, padding_mask 
       
    def _embed_MW(self, trg_MW):
        embedding = self.linear_embedding_MW(trg_MW)
        embedding = F.relu(embedding)            
        #embedding = embedding.unsqueeze(0)     
        padding_mask = torch.zeros((trg_MW.shape[0], 1), dtype=torch.bool, device=self.config.device)
        return embedding, padding_mask  

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.config.device)
    
    
    def forward(self, src_1H, mask_1H, src_13C, mask_13C, src_HSQC, mask_HSQC, src_COSY, mask_COSY, src_IR, mask_IR, src_MF, mask_MF, src_MS, mask_MS, trg_MW, trg_SMI_input=None):
        trg_MW = trg_MW.unsqueeze(1)        

        # Create embeddings
        embedding_src_1H, src_padding_mask_1H = self._embed_spectrum_1H(src_1H, mask_1H) if "1H" in self.config.training_mode else (None, None)
        current_batch_size = embedding_src_1H.shape[1] if "1H" in self.config.training_mode else False

        embedding_src_13C, src_padding_mask_13C = self._embed_spectrum_13C(src_13C, mask_13C) if "13C" in self.config.training_mode else (None, None)
        current_batch_size = embedding_src_13C.shape[1] if "13C" in self.config.training_mode else current_batch_size

        embedding_src_HSQC, src_padding_mask_HSQC = self._embed_spectrum_HSQC(src_HSQC, mask_HSQC) if "HSQC" in self.config.training_mode else (None, None)
        current_batch_size = embedding_src_HSQC.shape[1] if "HSQC" in self.config.training_mode  else current_batch_size

        embedding_src_COSY, src_padding_mask_COSY = self._embed_spectrum_COSY(src_COSY, mask_COSY) if "COSY" in self.config.training_mode else (None, None)
        current_batch_size = embedding_src_COSY.shape[1] if "COSY" in self.config.training_mode else current_batch_size

        embedding_src_IR, src_padding_mask_IR = self._embed_spectrum_IR(src_IR, mask_IR) if "IR" in self.config.training_mode else (None, None)       
        
        embedding_src_MF, src_padding_mask_MF = self._embed_MF(src_MF, mask_MF) if "MF" in self.config.training_mode else (None, None)

        embedding_src_MS, src_padding_mask_MS = self._embed_MS(src_MS, mask_MS) if "MS" in self.config.training_mode else (None, None)

        embedding_src_MW, src_padding_mask_MW = self._embed_MW(trg_MW) if "MW" in self.config.training_mode else (None, None)

        if embedding_src_IR != None:
            embedding_src_IR, src_padding_mask_IR = embedding_src_IR.unsqueeze(0), src_padding_mask_IR.unsqueeze(-1)

        if embedding_src_MW != None:
            embedding_src_MW = embedding_src_MW.unsqueeze(0) 


        feature_dim = 193 if "MS" in self.config.training_mode else 129
        feature_dim_IR = 130 if "MS" in self.config.training_mode else 66

        memory = []
        embedding_src = []
        src_padding_mask = []
        if embedding_src_1H is not None:
            embedding_src_1H, src_padding_mask_1H = self._create_embeddings_and_masks_1H(embedding_src_1H, src_padding_mask_1H, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )
            memory_1H = self.encoder_1H(embedding_src_1H, src_key_padding_mask=src_padding_mask_1H)
            memory.append(memory_1H)
            embedding_src.append(embedding_src_1H)
            src_padding_mask.append(src_padding_mask_1H)
        else:
            # create blank embeddings with blank mask 
            memory_1H = torch.zeros((feature_dim, current_batch_size, 128)).to(self.config.device)
            embedding_src_1H = torch.zeros((feature_dim, current_batch_size, 128)).to(self.config.device)
            src_padding_mask_1H = torch.ones((current_batch_size, feature_dim)).to(self.config.device)

            memory.append(memory_1H)
            embedding_src.append(embedding_src_1H)
            src_padding_mask.append(src_padding_mask_1H)            

        if embedding_src_13C is not None:
            embedding_src_13C, src_padding_mask_13C = self._create_embeddings_and_masks_13C(embedding_src_13C, src_padding_mask_13C, 
                                                                        embedding_src_MF, src_padding_mask_MF,
                                                                        embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                        embedding_src_MW, src_padding_mask_MW,
                                                                        )
            memory_13C = self.encoder_13C(embedding_src_13C, src_key_padding_mask=src_padding_mask_13C)
            memory.append(memory_13C)            
            embedding_src.append(embedding_src_13C)
            src_padding_mask.append(src_padding_mask_13C)
        else:
            # create blank embeddings with blank mask
            memory_13C = torch.zeros((feature_dim, current_batch_size, 128)).to(self.config.device)
            embedding_src_13C = torch.zeros((feature_dim, current_batch_size, 128)).to(self.config.device)
            src_padding_mask_13C = torch.ones((current_batch_size, feature_dim)).to(self.config.device)

            memory.append(memory_13C)
            embedding_src.append(embedding_src_13C)
            src_padding_mask.append(src_padding_mask_13C)   

        if embedding_src_HSQC is not None:
            embedding_src_HSQC, src_padding_mask_HSQC = self._create_embeddings_and_masks_HSQC(embedding_src_HSQC, src_padding_mask_HSQC, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )    
            memory_HSQC = self.encoder_HSQC(embedding_src_HSQC, src_key_padding_mask=src_padding_mask_HSQC)
            memory.append(memory_HSQC) 
            embedding_src.append(embedding_src_HSQC)
            src_padding_mask.append(src_padding_mask_HSQC)
        else:
            # create blank embeddings with blank mask 
            memory_HSQC = torch.zeros((feature_dim, current_batch_size, 128)).to(self.config.device)
            embedding_src_HSQC = torch.zeros((feature_dim, current_batch_size, 128)).to(self.config.device)
            src_padding_mask_HSQC = torch.ones((current_batch_size, feature_dim)).to(self.config.device)

            memory.append(memory_HSQC)
            embedding_src.append(embedding_src_HSQC)
            src_padding_mask.append(src_padding_mask_HSQC)  

        if embedding_src_COSY is not None:
            embedding_src_COSY, src_padding_mask_COSY = self._create_embeddings_and_masks_COSY(embedding_src_COSY, src_padding_mask_COSY, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )    
            memory_COSY = self.encoder_COSY(embedding_src_COSY, src_key_padding_mask=src_padding_mask_COSY)
            memory.append(memory_COSY) 
            embedding_src.append(embedding_src_COSY)
            src_padding_mask.append(src_padding_mask_COSY)
        else:
            # create blank embeddings with blank mask 
            memory_COSY = torch.zeros((65, current_batch_size, 128)).to(self.config.device)
            embedding_src_COSY = torch.zeros((65, current_batch_size, 128)).to(self.config.device)
            src_padding_mask_COSY = torch.ones((current_batch_size, 65)).to(self.config.device)

            memory.append(memory_COSY)
            embedding_src.append(embedding_src_COSY)
            src_padding_mask.append(src_padding_mask_COSY)  


        if embedding_src_IR is not None:
            embedding_src_IR, src_padding_mask_IR = self._create_embeddings_and_masks_IR(embedding_src_IR, src_padding_mask_IR, 
                                                                                embedding_src_MF, src_padding_mask_MF,
                                                                                embedding_src_MS, src_padding_mask_MS,                                                                            
                                                                                embedding_src_MW, src_padding_mask_MW,
                                                                                )
            memory_IR = self.encoder_IR(embedding_src_IR, src_key_padding_mask=src_padding_mask_IR)
            memory.append(memory_IR)  
            embedding_src.append(embedding_src_IR)
            src_padding_mask.append(src_padding_mask_IR)  
        else:
            # create blank embeddings with blank mask 
            memory_IR = torch.zeros((feature_dim_IR, current_batch_size, 128)).to(self.config.device)
            embedding_src_IR = torch.zeros((feature_dim_IR, current_batch_size, 128)).to(self.config.device)
            src_padding_mask_IR = torch.full((current_batch_size, feature_dim_IR), False, dtype=torch.bool).to(self.config.device)

            memory.append(memory_IR)
            embedding_src.append(embedding_src_IR)
            src_padding_mask.append(src_padding_mask_IR)                       

        memory = torch.cat(memory, dim=0) 
        embedding_src = torch.cat(embedding_src, dim=0) 
        src_padding_mask = torch.cat(src_padding_mask, dim=1) 
        memory = self.encoder_cross(memory, src_key_padding_mask=src_padding_mask)

        average_memory = torch.mean(memory, dim=0) 
        # Pass the average_memory to your linear layer (fp1)
        fingerprint = self.fp1(average_memory)        

        #import IPython; IPython.embed();
        
        if trg_SMI_input is None:
            return memory, embedding_src, src_padding_mask, fingerprint

        trg_seq_length, N = trg_SMI_input.shape
        trg_positions = (torch.arange(0, trg_seq_length)
                .unsqueeze(1)
                .expand(trg_seq_length, N)
                .to(self.config.device)) 

        embedding_trg = self.dropout2((self.embed_trg(trg_SMI_input) + self.pe_trg(trg_positions))).to(self.config.device) 
        trg_mask = self.generate_square_subsequent_mask(trg_seq_length).to(self.config.device)
        output_dec = self.decoder(embedding_trg, memory, tgt_mask=trg_mask, memory_key_padding_mask=src_padding_mask)

        if self.config.use_real_data:
            # Process with the real-data-specific pathway

            average_memory_ = self.real_data_linear(average_memory) 
            output_ = self.fc_out(output_dec) 
            expanded_avg_memory = average_memory_.unsqueeze(0).expand(output_.shape[0], -1, -1)
            output = (output_ + expanded_avg_memory) / 2

        else:
            output = self.fc_out(output_dec) 

        return output, fingerprint, memory, src_padding_mask
      

    

####################################################################################
############################# Multi GPU network ####################################
####################################################################################

class TransformerMultiGPU(pl.LightningModule):

    def __init__(self, config):
        super(TransformerMultiGPU, self).__init__()
        #self.l1 = torch.nn.Linear(512, 512)
        self.start_time = time.time()
        self.config=config
        os.makedirs(self.config.model_save_dir, exist_ok=True)        
        
        # Make dataloaders, tokenizer, model, optimizer, loss_fn 
        SMI_loss_fn, MF_loss_fn, MW_loss_fn, FP_loss_fn, itos, stoi, itos_MF, stoi_MF, sgnn_means_stds = make_pl(config)
        model = MultimodalTransformer(config, src_pad_idx=0)

        self.model = model
        self.SMI_loss_fn = SMI_loss_fn
        self.MF_loss_fn = MF_loss_fn
        self.FP_loss_fn = FP_loss_fn
        self.MW_loss_fn = MW_loss_fn
        
        if config.training_setup == "pretraining":
            self.lr = config.lr_pretraining
        elif config.training_setup == "finetuning":
            self.lr = config.lr_finetuning
            
        self.itos = itos
        self.stoi = stoi
        self.itos_MF = itos_MF
        self.stoi_MF = stoi_MF        
        self.sgnn_means_stds = sgnn_means_stds


    def forward(self, x):
        src_1H = x['src_1H'] if "1H" in self.config.training_mode else None
        mask_1H = x['mask_1H'] if "1H" in self.config.training_mode else None
        src_13C = x['src_13C'] if "13C" in self.config.training_mode else None
        mask_13C = x['mask_13C'] if "13C" in self.config.training_mode else None
        src_HSQC = x['src_HSQC'] if "HSQC" in self.config.training_mode else None
        mask_HSQC = x['mask_HSQC'] if "HSQC" in self.config.training_mode else None
        src_COSY = x['src_COSY'] if "COSY" in self.config.training_mode else None
        mask_COSY = x['mask_COSY'] if "COSY" in self.config.training_mode else None
        src_IR = x['src_IR'] if "IR" in self.config.training_mode else None
        mask_IR = x['mask_IR'] if "IR" in self.config.training_mode else None
        src_MF = x['src_MF'] if "MF" in self.config.training_mode else None
        mask_MF = x['mask_MF'] if "MF" in self.config.training_mode else None
        src_MS = x['src_MS'] if "MS" in self.config.training_mode else None
        mask_MS = x['mask_MS'] if "MS" in self.config.training_mode else None      
        trg_MW = x['trg_MW'] if "MW" in self.config.training_mode else None    
        trg_enc_SMI = x['trg_enc_SMI']
        trg_FP = x['trg_FP']
        trg_SMI_input = x['trg_SMI_input']
        #import IPython; IPython.embed();

        gen_output, gen_FP, memory, src_padding_mask = self.model(src_1H, mask_1H, 
                                                                src_13C, mask_13C, 
                                                                src_HSQC, mask_HSQC, 
                                                                src_COSY, mask_COSY, 
                                                                src_IR, mask_IR, 
                                                                src_MF, mask_MF, 
                                                                src_MS, mask_MS, 
                                                                trg_MW, trg_SMI_input)



        return gen_output, gen_FP, memory, src_padding_mask 

    
    def training_step(self, x, batch_idx): 

        trg_enc_SMI = x['trg_enc_SMI']
        #src_MF = x['src_MF']
        trg_FP = x['trg_FP']
        trg_MW = x['trg_MW']
        src_HSQC = x['src_HSQC'] if "HSQC" in self.config.training_mode else None        
        
        #trg_SMI = trg_SMI.squeeze(0)
        #print(trg_SMI.shape)
        trg_enc_SMI = trg_enc_SMI.transpose(0, 1)
        trg_SMI_input = trg_enc_SMI[:-1, :] # Remove <EOS> token from target sequence
        x['trg_SMI_input'] = trg_SMI_input
        gen_output, gen_FP, memory, src_padding_mask = self.forward(x)
        ### Learning task Smiles generation
        temperature = 1
        max_output_len = self.config.max_len
        if self.config.gen_SMI_sequence==True:

            gen_enc_SMI, gen_conv_SMI, confidence_list = self._generate_sequences(self.model, 
                                                            self.config, 
                                                            memory, 
                                                            src_padding_mask, 
                                                            max_output_len, 
                                                            temperature, 
                                                            self.stoi, 
                                                            self.itos)   
            #import IPython; IPython.embed();
        else:
            gen_enc_SMI=None 
            gen_conv_SMI=None
            confidence_list=None

        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:,:], self.itos)

        losses_list, MW_loss, validity_term, count_reward, tanimoto_mean, avg_sim_error = calculate_total_loss(
                                                                        gen_output, 
                                                                        trg_enc_SMI, 
                                                                        gen_FP, 
                                                                        trg_FP, 
                                                                        gen_conv_SMI,
                                                                        trg_conv_SMI, 
                                                                        trg_MW,
                                                                        self.FP_loss_fn, 
                                                                        self.SMI_loss_fn, 
                                                                        self.MW_loss_fn,
                                                                        self.sgnn_means_stds,    
                                                                        src_HSQC,
                                                                        self.config,
                                                                        batch_idx)


        #### update this part
        tensorboard_logs = {'train_loss': losses_list[0]}
        self.logger.experiment.log({'train_loss': losses_list[0]})
        self.logger.experiment.log({'SMI_loss': losses_list[1]})
        self.logger.experiment.log({'FP_loss': losses_list[2]})
        self.logger.experiment.log({'weight_loss': losses_list[3]})
        self.logger.experiment.log({'sgnn_loss': losses_list[4]})
        self.logger.experiment.log({'tanimoto_loss': losses_list[5]})
        self.logger.experiment.log({'valitity_loss': losses_list[6]})
        self.logger.experiment.log({'MW_loss': MW_loss})
        self.logger.experiment.log({'validity_term': validity_term})
        self.logger.experiment.log({'count_reward': count_reward})
        self.logger.experiment.log({'tanimoto_mean': tanimoto_mean})
        self.logger.experiment.log({'avg_sim_error': avg_sim_error})
        
        return {'loss': losses_list[0], 'log': tensorboard_logs}
    

    def validation_step(self, x, batch_idx):
        trg_enc_SMI = x['trg_enc_SMI']
        #src_MF = x['src_MF']
        trg_FP = x['trg_FP']
        trg_MW = x['trg_MW']
        src_HSQC = x['src_HSQC'] if "HSQC" in self.config.training_mode else None              
        #print(trg_SMI.shape)

        #trg_SMI = trg_SMI.squeeze(0)  ### Do i need that I didn't have it here before?
        trg_enc_SMI = trg_enc_SMI.transpose(0, 1)
        trg_SMI_input = trg_enc_SMI[:-1, :] # Remove <EOS> token from target sequence
        x['trg_SMI_input'] = trg_SMI_input

        gen_output, gen_FP, memory, src_padding_mask = self.forward(x)
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:,:], self.itos)
        #import IPython; IPython.embed();

        ### Learning task Smiles generation
        temperature = 1
        max_output_len = self.config.max_len        
        if self.config.gen_SMI_sequence==True:
            gen_enc_SMI, gen_conv_SMI, confidence_list = self._generate_sequences(self.model, 
                                                              self.config, 
                                                              memory, 
                                                              src_padding_mask, 
                                                              max_output_len, 
                                                              temperature, 
                                                              self.stoi, 
                                                              self.itos)  
        else:
            gen_enc_SMI=None 
            gen_conv_SMI=None
            confidence_list=None


        losses_list, MW_loss, validity_term, count_reward, tanimoto_mean, avg_sim_error = calculate_total_loss(
                                                                          gen_output, 
                                                                          trg_enc_SMI, 
                                                                          gen_FP, 
                                                                          trg_FP, 
                                                                          gen_conv_SMI,
                                                                          trg_conv_SMI, 
                                                                          trg_MW,
                                                                          self.FP_loss_fn , 
                                                                          self.SMI_loss_fn , 
                                                                          self.MW_loss_fn ,
                                                                          self.sgnn_means_stds,    
                                                                          src_HSQC,
                                                                          self.config,
                                                                          batch_idx)


        #### update this part
        tensorboard_logs = {'test_loss': losses_list[0]}   ### test_loss
        self.logger.experiment.log({'test_loss': losses_list[0]})   ### test_loss
        self.logger.experiment.log({'SMI_loss': losses_list[1]})
        self.logger.experiment.log({'FP_loss': losses_list[2]})
        self.logger.experiment.log({'weight_loss': losses_list[3]})
        self.logger.experiment.log({'sgnn_loss': losses_list[4]})
        self.logger.experiment.log({'tanimoto_loss': losses_list[5]})
        self.logger.experiment.log({'valitity_loss': losses_list[6]})
        self.logger.experiment.log({'MW_loss': MW_loss})
        self.logger.experiment.log({'validity_term': validity_term})
        self.logger.experiment.log({'count_reward': count_reward})
        self.logger.experiment.log({'tanimoto_mean': tanimoto_mean})
        self.logger.experiment.log({'avg_sim_error': avg_sim_error})

        return {'loss': losses_list[0], 'log': tensorboard_logs}
    
    def on_epoch_end(self):
        epoch = self.current_epoch
        loss = self.trainer.callback_metrics.get('train_loss', None)  
        save_path = os.path.join(self.config.model_save_dir, f"MultimodalTransformer_Epoch_{epoch}_Loss{loss:.3f}.pth")
        torch.save(self.model.state_dict(), save_path)
        
    def on_batch_end(self):
        elapsed_time = time.time() - self.start_time
        time_obj = datetime.timedelta(seconds=elapsed_time)
        time_str = str(time_obj).split('.')[0]  # This will give you a string formatted as 'H:M:S'


        if elapsed_time  >=  self.config.model_save_interval:  
            epoch = self.current_epoch
            loss = self.trainer.callback_metrics.get('train_loss', None)  
            save_path = os.path.join(self.config.model_save_dir, f"MultimodalTransformer_time_{str(time.time())}_Loss_{loss:.3f}.pth")
            ckp_save_path = os.path.join(self.config.model_save_dir, f"MultimodalTransformer_time_{str(time.time())}_Loss_{loss:.3f}.ckpt")
            torch.save(self.model.state_dict(), save_path)
            self.trainer.save_checkpoint(ckp_save_path)
            print(f'Model saved at {save_path}')
            self.start_time = time.time()

        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        return {'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler,
                             'monitor': 'loss',  # Metric to monitor for lr scheduling
                            }
                }            
            
    def _generate_sequences(self, model, config, memory, src_padding_mask, max_output_len, temperature, stoi, itos):
        #memory = model.transformer.encoder(src, src_key_padding_mask=src_padding_mask)
        model.eval()
        with torch.no_grad():
            # Initialize target tensor with the start token index
            N = memory.size(1)
            gen_tensor1 = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
            gen_tensor2 = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
            confidence_list_greedy = []
            confidence_list_multinom =[]
            for _ in range(max_output_len):
                gen_seq_length, N = gen_tensor1.shape
                gen_positions = (
                    torch.arange(0, gen_seq_length)
                    .unsqueeze(1)
                    .expand(gen_seq_length, N)
                    .to(config.device)
                )

                embedding_gen = model.dropout2((model.embed_trg(gen_tensor1) + model.pe_trg(gen_positions)))
                gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

                output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
                output = model.fc_out(output)
                probabilities = F.softmax(output / temperature, dim=2)

                if config.sampling_method in ["greedy", "mix"]:

                    next_word1 = torch.argmax(probabilities[-1, :, :], dim=1)

                    # calculate probability of sequence
                    max_prob = probabilities[-1, :, :].gather(1, next_word1.unsqueeze(-1)).squeeze()  # Get the probability of the predicted token
                    confidence_list_greedy.append(max_prob)  # Append to list
                    next_word1 = next_word1.unsqueeze(0)
                    gen_tensor1 = torch.cat((gen_tensor1, next_word1), dim=0)

                if config.sampling_method in ["multinomial", "mix"]:

                    next_word2 = torch.multinomial(probabilities[-1, :, :], 1)  
                    sel_prob = probabilities[-1, :, :].gather(1, next_word2).squeeze()  # Get the probability of the predicted token
                    confidence_list_multinom.append(sel_prob)
                    next_word2 = next_word2.squeeze(1).unsqueeze(0)
                    gen_tensor2 = torch.cat((gen_tensor2, next_word2), dim=0)

        # print("generate_sequences")    
        # Convert tensor to SMILES strings
        model.train()
        if config.sampling_method == "greedy":
            confidence_list_greedy = torch.stack(confidence_list_greedy)
            if len(confidence_list_greedy.shape) ==1:
                confidence_list_greedy = confidence_list_greedy.unsqueeze(-1)            
            #gen_smi1, confidence_list_greedy = hf.tensor_to_smiles_and_prob(gen_tensor1[1:], confidence_list_greedy, itos)
            gen_smi1, confidence_list_greedy = hf.tensor_to_smiles_and_prob_2(gen_tensor1[1:], confidence_list_greedy, itos)
            return gen_tensor1, gen_smi1, confidence_list_greedy

        elif config.sampling_method == "multinomial":
            confidence_list_multinom = torch.stack(confidence_list_multinom)
            if len(confidence_list_multinom.shape) ==1:
                confidence_list_multinom = confidence_list_multinom.unsqueeze(-1)              
            #gen_smi2, confidence_list_multinom = hf.tensor_to_smiles_and_prob(gen_tensor2[1:], confidence_list_multinom, itos)
            gen_smi2, confidence_list_multinom = hf.tensor_to_smiles_and_prob_2(gen_tensor2[1:], confidence_list_multinom, itos)
            return gen_tensor2, gen_smi2, confidence_list_multinom

        elif config.sampling_method == "mix":
            confidence_list_multinom = torch.stack(confidence_list_multinom)
            confidence_list_greedy = torch.stack(confidence_list_greedy)
            if len(confidence_list_greedy.shape) ==1:
                confidence_list_greedy = confidence_list_greedy.unsqueeze(-1)
            if len(confidence_list_multinom.shape) ==1:
                confidence_list_multinom = confidence_list_multinom.unsqueeze(-1)                
            #gen_smi1, confidence_list_greedy = hf.tensor_to_smiles_and_prob(gen_tensor1[1:], confidence_list_greedy, itos)
            #gen_smi2, confidence_list_multinom = hf.tensor_to_smiles_and_prob(gen_tensor2[1:], confidence_list_multinom, itos)
            gen_smi1, confidence_list_greedy = hf.tensor_to_smiles_and_prob_2(gen_tensor1[1:], confidence_list_multinom, itos)
            gen_smi2, confidence_list_multinom = hf.tensor_to_smiles_and_prob_2(gen_tensor2[1:], confidence_list_multinom, itos)            
            gen_tensor, gen_smi, confidence_list = hf.combine_gen_sims(gen_smi1, gen_tensor1, confidence_list_greedy, gen_smi2, gen_tensor2, confidence_list_multinom)
            return gen_tensor, gen_smi, confidence_list

