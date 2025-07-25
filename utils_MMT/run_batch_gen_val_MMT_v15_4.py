# Standard library imports
import operator
import random
import statistics
from collections import defaultdict
from functools import reduce

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm  # for progress bar
import matplotlib.pyplot as plt
import pandas as pd


# Custom utility/module imports
from utils_MMT.dataloaders_pl_v15_4 import MultimodalData, collate_fn

from utils_MMT.models_MMT_v15_4 import MultimodalTransformer, TransformerMultiGPU
from utils_MMT.models_CLIP_v15_4 import CLIPModel, ChembertaFingerprint, CLIPMultiGPU, CLIP_make   ############# CHANGE WHEN I HAVE A CLIP V8 model trained

import utils_MMT.sgnn_code_pl_v15_4 as sgnn
import utils_MMT.train_test_functions_pl_v15_4 as ttf
import utils_MMT.validate_generate_MMT_v15_4 as vgmmt
import utils_MMT.helper_functions_pl_v15_4 as hf


# Constants and Random Seed Initialization
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


#Load SGNN mean values
graph_representation = "sparsified"
target = "13C"
#train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H = None, None, None, None
train_y_mean_C, train_y_std_C = sgnn.load_std_mean(target,graph_representation)
target = "1H"
train_y_mean_H, train_y_std_H = sgnn.load_std_mean(target,graph_representation)
sgnn_means_stds = (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2**32))
    random.seed(torch.initial_seed() % (2**32))
    
##3change batch
def load_data_and_CLIP_model(config, mode, stoi, stoi_MF, itos, itos_MF, single=True):
    
    # Loading the test dataset and dataloader
    val_data = MultimodalData(config, stoi, stoi_MF, mode=mode)
    if single:
        val_dataloader = DataLoader(val_data, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    collate_fn=collate_fn, 
                                    drop_last=True,
                                    worker_init_fn=worker_init_fn)
    else:
        val_dataloader = DataLoader(val_data, 
                                    batch_size=config.batch_size, 
                                    shuffle=False, 
                                    collate_fn=collate_fn, 
                                    drop_last=True,
                                    worker_init_fn=worker_init_fn)        
    
    CLIP_multi_gpu_model = CLIPMultiGPU(config)
    checkpoint_path = config.CLIP_model_path
    CLIP_model = CLIP_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)

    #CLIP_model, optimizer = CLIP_make(config, stoi, stoi_MF, itos)
    CLIP_model.to(config.device)
    return CLIP_model, val_dataloader
#model, val_dataloader = load_data_and_model(config)
#data_dict = next(iter(val_dataloader))



# Function to duplicate tensor
def duplicate_tensor(tensor, n_times):
    repeat_dims = [n_times] + [1] * (tensor.dim() - 1)
    duplicated_tensor = tensor.repeat(*repeat_dims)
    return duplicated_tensor.to("cuda")

# Function to duplicate tensor
def duplicate_dict(data_dict, n_times):
    # Dictionary to hold duplicated tensors
    duplicated_dict = {}
    # Duplicating tensors within each key of the dictionary
    for key, tensor in data_dict.items():
        duplicated_tensor = duplicate_tensor(tensor, n_times)
        duplicated_dict[key] = duplicated_tensor

    return duplicated_dict

# Function to filter valid SMILES
def filter_valid_smiles_and_canonicolize(smiles_list, canonical=True, isomericSmiles=False):
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            gen_smi = Chem.MolToSmiles(mol, canonical=canonical, doRandom=False, isomericSmiles=isomericSmiles)
            valid_smiles.append(gen_smi)
    return valid_smiles


# Multinomial
def multinomial_sequence_multi(model, memory, src_padding_mask, stoi, config):
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
    multinom_token_prob = torch.stack(multinom_token_prob) 

    # remove "SOS" token
    multinom_tensor = multinom_tensor[1:,:]

    return multinom_tensor, multinom_token_prob


def add_tanimoto_similarity(trg_conv_SMI, combined_list):
    # Generate fingerprint for the ground truth molecule
    ground_truth_mol = Chem.MolFromSmiles(trg_conv_SMI)
    ground_truth_fp = AllChem.GetMorganFingerprintAsBitVect(ground_truth_mol, 2, nBits=512)
    # Calculate Tanimoto similarity and add to the list
    for item in combined_list:
        smiles = item[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Check if the molecule is valid
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(ground_truth_fp, fp)
            item.append(tanimoto_similarity)
    return combined_list
#print(combined_list)


def calculate_molecular_weights(smiles_list):
    molecular_weights = []  # List to store the molecular weights
    for smiles in tqdm(smiles_list):  # tqdm displays a progress bar
        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to Mol object
        if mol is not None:  # Check if the conversion was successful
            mw = Descriptors.ExactMolWt(mol)  # Calculate molecular weight
            molecular_weights.append(mw)
        else:
            print(f"Failed to convert SMILES: {smiles}")
            molecular_weights.append(None)  # Append None for failed conversions
    return molecular_weights

def prepare_HSQC_data_from_src(src_HSQC_list):
    tensor_HSQC = []
    for src_HSQC in src_HSQC_list:
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
    return tensor_HSQC

def prepare_COSY_data_from_src(src_COSY_list):
    tensor_COSY = []
    for src_COSY in src_COSY_list:
        for src in src_COSY:
            # Find the rows where both elements are not zero
            mask = (src != 0).all(dim=1)
            # Use boolean indexing to get the rows
            filtered_tensor = src[mask]
            if list(filtered_tensor) != []:
                list_tensors = [filtered_tensor[:,0]*10, filtered_tensor[:,1]*10]
                reshaped_tensors = [tensor.unsqueeze(1) for tensor in list_tensors]
                # Concatenate the tensors along the second dimension (columns)
                combined_tensor = torch.cat(reshaped_tensors, dim=1)
                tensor_COSY.append(combined_tensor)
            else:
                tensor_COSY.append([])
    return tensor_COSY

def add_HSQC_COSY_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, _trg_conv_SMI, n_times):
    src_HSQC_list = [data_dict_dup["src_HSQC"]]
    src_COSY_list = [data_dict_dup["src_COSY"]]
    tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
    tensor_COSY = prepare_COSY_data_from_src(src_COSY_list)

    trg_conv_SMI_list = [_trg_conv_SMI for _ in range(n_times)]

    trg_MW = [rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(_trg_conv_SMI)) for _ in range(n_times)]

    #trg_conv_SMI_list = [_trg_conv_SMI for i in range(n_times)]
    #trg_MW = [rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(_trg_conv_SMI)) for i in range(n_times)]
    #import IPython; IPython.embed();

    tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis = hf.calculate_tanimoto_and_mol_weights(gen_conv_SMI_list, trg_conv_SMI_list, trg_MW)

    avg_sim_error_HSQC, avg_sim_error_COSY, sim_error_list_HSQC, sim_error_list_COSY, batch_data = ttf.run_sgnn_sim_calculations_if_possible_return_spectra(df_succ_smis, tensor_HSQC, tensor_COSY, sgnn_means_stds, config)

    # Add numbers as the fourth item to each inner list
    for i, inner_list in enumerate(combined_list):
        try:
            inner_list.append([sim_error_list_HSQC[i], sim_error_list_COSY[i]])
        except:
            inner_list.append(9)

    return combined_list, batch_data




# def run_model(model, data_dict, config):
#     # Model preparation and data loading for GPU
    
#     x = data_dict
#     src_1H = x['src_1H'].to(config.device) if "1H" in config.training_mode else None
#     mask_1H = x['mask_1H'].to(config.device) if "1H" in config.training_mode else None
#     src_13C = x['src_13C'].to(config.device) if "13C" in config.training_mode else None
#     mask_13C = x['mask_13C'].to(config.device) if "13C" in config.training_mode else None
#     src_HSQC = x['src_HSQC'].to(config.device) if "HSQC" in config.training_mode else None
#     mask_HSQC = x['mask_HSQC'].to(config.device) if "HSQC" in config.training_mode else None
#     src_COSY = x['src_COSY'].to(config.device) if "COSY" in config.training_mode else None
#     mask_COSY = x['mask_COSY'].to(config.device) if "COSY" in config.training_mode else None
#     src_IR = x['src_IR'].to(config.device) if "IR" in config.training_mode else None
#     mask_IR = x['mask_IR'].to(config.device) if "IR" in config.training_mode else None
#     src_MF = x['src_MF'].to(config.device) if "MF" in config.training_mode else None
#     mask_MF = x['mask_MF'].to(config.device) if "MF" in config.training_mode else None
#     src_MS = x['src_MS'].to(config.device) if "MS" in config.training_mode else None
#     mask_MS = x['mask_MS'].to(config.device) if "MS" in config.training_mode else None      
#     trg_MW = x['trg_MW'].to(config.device) if "MW" in config.training_mode else None    
#     trg_enc_SMI = x['trg_enc_SMI'].to(config.device) 

    
#     trg_MW = trg_MW.unsqueeze(1)        
#     with torch.no_grad():
#         # Create embeddings
#         embedding_src_1H, src_padding_mask_1H = model._embed_spectrum_1H(src_1H, mask_1H) if "1H" in config.training_mode else (None, None)
#         current_batch_size = embedding_src_1H.shape[1] if "1H" in config.training_mode else False

#         embedding_src_13C, src_padding_mask_13C = model._embed_spectrum_13C(src_13C, mask_13C) if "13C" in config.training_mode else (None, None)
#         current_batch_size = embedding_src_13C.shape[1] if "13C" in config.training_mode else current_batch_size

#         embedding_src_HSQC, src_padding_mask_HSQC = model._embed_spectrum_HSQC(src_HSQC, mask_HSQC) if "HSQC" in config.training_mode else (None, None)
#         current_batch_size = embedding_src_HSQC.shape[1] if "HSQC" in config.training_mode  else current_batch_size

#         embedding_src_COSY, src_padding_mask_COSY = model._embed_spectrum_COSY(src_COSY, mask_COSY) if "COSY" in config.training_mode else (None, None)
#         current_batch_size = embedding_src_COSY.shape[1] if "COSY" in config.training_mode else current_batch_size

#         embedding_src_IR, src_padding_mask_IR = model._embed_spectrum_IR(src_IR, mask_IR) if "IR" in config.training_mode else (None, None)       

#         embedding_src_MF, src_padding_mask_MF = model._embed_MF(src_MF, mask_MF) if "MF" in config.training_mode else (None, None)

#         embedding_src_MS, src_padding_mask_MS = model._embed_MS(src_MS, mask_MS) if "MS" in config.training_mode else (None, None)

#         embedding_src_MW, src_padding_mask_MW = model._embed_MW(trg_MW) if "MW" in config.training_mode else (None, None)

#         if embedding_src_IR != None:
#             embedding_src_IR, src_padding_mask_IR = embedding_src_IR.unsqueeze(0), src_padding_mask_IR.unsqueeze(-1)

#         if embedding_src_MW != None:
#             embedding_src_MW = embedding_src_MW.unsqueeze(0) 

#         feature_dim = 193 if "MS" in config.training_mode else 129
#         feature_dim_IR = 130 if "MS" in config.training_mode else 66

#         memory = []
#         embedding_src = []
#         src_padding_mask = []
#         if embedding_src_1H is not None:
#             embedding_src_1H, src_padding_mask_1H = model._create_embeddings_and_masks_1H(embedding_src_1H, src_padding_mask_1H, 
#                                                                                 embedding_src_MF, src_padding_mask_MF,
#                                                                                 embedding_src_MS, src_padding_mask_MS,                                                                            
#                                                                                 embedding_src_MW, src_padding_mask_MW,
#                                                                                 )
#             memory_1H = model.encoder_1H(embedding_src_1H, src_key_padding_mask=src_padding_mask_1H)
#             memory.append(memory_1H)
#             embedding_src.append(embedding_src_1H)
#             src_padding_mask.append(src_padding_mask_1H)
#         else:
#             # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
#             memory_1H = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
#             embedding_src_1H = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
#             src_padding_mask_1H = torch.ones((current_batch_size, feature_dim)).to(config.device)

#             memory.append(memory_1H)
#             embedding_src.append(embedding_src_1H)
#             src_padding_mask.append(src_padding_mask_1H)
            
#         if embedding_src_13C is not None:
#             embedding_src_13C, src_padding_mask_13C = model._create_embeddings_and_masks_13C(embedding_src_13C, src_padding_mask_13C, 
#                                                                         embedding_src_MF, src_padding_mask_MF,
#                                                                         embedding_src_MS, src_padding_mask_MS,                                                                            
#                                                                         embedding_src_MW, src_padding_mask_MW,
#                                                                         )
#             memory_13C = model.encoder_13C(embedding_src_13C, src_key_padding_mask=src_padding_mask_13C)
#             memory.append(memory_13C)            
#             embedding_src.append(embedding_src_13C)
#             src_padding_mask.append(src_padding_mask_13C)
#         else:
#             # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
#             memory_13C = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
#             embedding_src_13C = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
#             src_padding_mask_13C = torch.ones((current_batch_size, feature_dim)).to(config.device)

#             memory.append(memory_13C)
#             embedding_src.append(embedding_src_13C)
#             src_padding_mask.append(src_padding_mask_13C)   
            
#         if embedding_src_HSQC is not None:
#             embedding_src_HSQC, src_padding_mask_HSQC = model._create_embeddings_and_masks_HSQC(embedding_src_HSQC, src_padding_mask_HSQC, 
#                                                                                 embedding_src_MF, src_padding_mask_MF,
#                                                                                 embedding_src_MS, src_padding_mask_MS,                                                                            
#                                                                                 embedding_src_MW, src_padding_mask_MW,
#                                                                                 )    
#             memory_HSQC = model.encoder_HSQC(embedding_src_HSQC, src_key_padding_mask=src_padding_mask_HSQC)
#             memory.append(memory_HSQC) 
#             embedding_src.append(embedding_src_HSQC)
#             src_padding_mask.append(src_padding_mask_HSQC)
#         else:
#             # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
#             memory_HSQC = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
#             embedding_src_HSQC = torch.zeros((feature_dim, current_batch_size, 128)).to(config.device)
#             src_padding_mask_HSQC = torch.ones((current_batch_size, feature_dim)).to(config.device)

#             memory.append(memory_HSQC)
#             embedding_src.append(embedding_src_HSQC)
#             src_padding_mask.append(src_padding_mask_HSQC) 
            
#         if embedding_src_COSY is not None:
#             embedding_src_COSY, src_padding_mask_COSY = model._create_embeddings_and_masks_COSY(embedding_src_COSY, src_padding_mask_COSY, 
#                                                                                 embedding_src_MF, src_padding_mask_MF,
#                                                                                 embedding_src_MS, src_padding_mask_MS,                                                                            
#                                                                                 embedding_src_MW, src_padding_mask_MW,
#                                                                                 )    
#             memory_COSY = model.encoder_COSY(embedding_src_COSY, src_key_padding_mask=src_padding_mask_COSY)
#             memory.append(memory_COSY) 
#             embedding_src.append(embedding_src_COSY)
#             src_padding_mask.append(src_padding_mask_COSY)
#         else:
#             # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
#             memory_COSY = torch.zeros((65, current_batch_size, 128)).to(config.device)
#             embedding_src_COSY = torch.zeros((65, current_batch_size, 128)).to(config.device)
#             src_padding_mask_COSY = torch.ones((current_batch_size, 65)).to(config.device)

#             memory.append(memory_COSY)
#             embedding_src.append(embedding_src_COSY)
#             src_padding_mask.append(src_padding_mask_COSY) 
            
#         if embedding_src_IR is not None:
#             embedding_src_IR, src_padding_mask_IR = model._create_embeddings_and_masks_IR(embedding_src_IR, src_padding_mask_IR, 
#                                                                                 embedding_src_MF, src_padding_mask_MF,
#                                                                                 embedding_src_MS, src_padding_mask_MS,                                                                            
#                                                                                 embedding_src_MW, src_padding_mask_MW,
#                                                                                 )
#             memory_IR = model.encoder_IR(embedding_src_IR, src_key_padding_mask=src_padding_mask_IR)
#             memory.append(memory_IR)  
#             embedding_src.append(embedding_src_IR)
#             src_padding_mask.append(src_padding_mask_IR)                          
#         else:
#             # create blank embeddings with blank mask For embedding_src_1H and memory_1H src_padding_mask_1H
#             memory_IR = torch.zeros((feature_dim_IR, current_batch_size, 128)).to(config.device)
#             embedding_src_IR = torch.zeros((feature_dim_IR, current_batch_size, 128)).to(config.device)
#             src_padding_mask_IR = torch.full((current_batch_size, feature_dim_IR), False, dtype=torch.bool).to(config.device)

#             memory.append(memory_IR)
#             embedding_src.append(embedding_src_IR)
#             src_padding_mask.append(src_padding_mask_IR)  

#         memory = torch.cat(memory, dim=0) 
#         # import IPython; IPython.embed();
#         embedding_src = torch.cat(embedding_src, dim=0) 
#         src_padding_mask = torch.cat(src_padding_mask, dim=1) 
#         memory = model.encoder_cross(memory, src_key_padding_mask=src_padding_mask)  #########################################
#         average_memory = torch.mean(memory, dim=0) 

#         fingerprint = model.fp1(average_memory)        
                   
#         #print(src_padding_mask.shape)
#         #import IPython; IPython.embed();
#     return memory, src_padding_mask, trg_enc_SMI.to(config.device), fingerprint, src_HSQC


# Number of times to duplicate each tensor

def run_MMT_generations(config, MMT_model, CLIP_model, dataloader, stoi, itos, sort_num):
    """ sort_num -> 3 = taniomoto, 4 = HSQC error, 3 = Cosine sim, 2 = Clip Loss
    just runs Multinomial sampling
    """
    results_dict = defaultdict(list)
    # generate all the smiles of trg and greedy gen
    print("Start Dataloader")
    for i, data_dict in tqdm(enumerate(dataloader)):
        gen_conv_SMI_list = []

        trg_enc_SMI = data_dict["trg_enc_SMI"][0].to(config.device)
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        ### multiply the input to paralellize the generation
        if config.MMT_generations < config.MMT_batch:
            config.MMT_batch = config.MMT_generations 
        data_dict_dup = duplicate_dict(data_dict, config.MMT_batch)

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(MMT_model,
                                                                       data_dict_dup, 
                                                                       config)
        counter = 0
        ### Here I increase the temperature in case if it does not generate enough diverse molecules but sets it back to the original
        ### value after enough molecules were found.
        temp_orig = config.temperature
        while len(gen_conv_SMI_list)<config.MMT_generations:
            if counter%10==0:
                config.temperature = config.temperature + 0.1
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi(MMT_model, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(multinom_tensor, multinom_token_prob, itos)
            gen_conv_SMI = filter_valid_smiles_and_canonicolize(gen_conv_SMI)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            gen_conv_SMI_list = list(set(gen_conv_SMI_list))
            counter += 1
        print(config.temperature)
        config.temperature = temp_orig

        gen_conv_SMI_list = gen_conv_SMI_list[:config.MMT_batch]
        
        mean_loss, losses, logits, targets, dot_similarity= CLIP_model.inference(data_dict_dup, gen_conv_SMI_list)

        combined_list = [[smile, num.item(), cosim.item()] for smile, num, cosim in zip(gen_conv_SMI_list, losses, dot_similarity)]
        ### Sort by the lowest similarity
        #sorted_list = sorted(combined_list, key=lambda x: x[1])


        combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
        combined_list, batch_data = add_HSQC_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.MMT_batch)
        #import IPython; IPython.embed();
        sorted_list = sorted(combined_list, key=lambda x: -x[sort_num]) # sort by highest tanimoto similarity

        results_dict[trg_conv_SMI] = [sorted_list, batch_data]
        if i == config.n_samples:
            break
    return results_dict
    
def run_test_performance_CLIP(config, 
                         stoi, 
                         stoi_MF, 
                         itos, 
                         itos_MF):

    mode = "val"
    CLIP_model, val_dataloader = load_data_and_CLIP_model(config, 
                                                            mode, 
                                                            stoi, 
                                                            stoi_MF, 
                                                            itos, 
                                                            itos_MF)
    
    model_MMT = CLIP_model.CLIP_model.MT_model
    model_MMT, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF)    ################# CHANGE WHEN FINAL MMT WITH CLIP IS WORKING
    model_CLIP = CLIP_model.CLIP_model
    
    # Number of times to duplicate each tensor
    n_times = config.multinom_runs
    temp_orig = config.temperature

    results_dict = defaultdict(list)
    # generate all the smiles of trg and greedy gen
    for i, data_dict in tqdm(enumerate(val_dataloader)):
        gen_conv_SMI_list = []

        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        ### multiply the input to paralellize the generation
        data_dict_dup = duplicate_dict(data_dict, n_times)  # Maybe should hardcode it here as 64 - it will always cut it down to the number needed with ntimes

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                       data_dict_dup, 
                                                                       config)
        counter = 0
        ### Here I increase the temperature in case if it does not generate enough diverse molecules but sets it back to the original
        ### value after enough molecules were found.
        while len(gen_conv_SMI_list) < n_times:
            # increase the temperature if not enough different molecules get generated
            if counter%10==0:
                config.temperature = config.temperature + 0.1
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi(model_MMT, memory, src_padding_mask, stoi, config)
            #import IPython; IPython.embed();
            if n_times == 1:
                 multinom_token_prob = multinom_token_prob.unsqueeze(1)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(multinom_tensor, multinom_token_prob, itos)
            gen_conv_SMI = filter_valid_smiles_and_canonicolize(gen_conv_SMI)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            gen_conv_SMI_list = list(set(gen_conv_SMI_list))
            counter += 1
        print(config.temperature)
        config.temperature = temp_orig

        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]

        mean_loss, losses, logits, targets, dot_similarity= model_CLIP.inference(data_dict_dup, gen_conv_SMI_list)
        combined_list = [[smile, num.item(), cosim.item()] for smile, num, cosim in zip(gen_conv_SMI_list, losses, dot_similarity)]
        ### Sort by the lowest similarity
        #sorted_list = sorted(combined_list, key=lambda x: x[1])
        #import IPython; IPython.embed();
        combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
        combined_list, batch_data = add_HSQC_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.multinom_runs) # config.MMT_batch

        sorted_list = sorted(combined_list, key=lambda x: -x[3]) # SMILES = 0, losses =1, dot_sim= 2, tanimoto = 3 

        results_dict[trg_conv_SMI] = [sorted_list, batch_data]
        #if i == config.n_samples:
        #    break
    return results_dict
    

def run_test_performance_CLIP_greedy(config, 
                         stoi, 
                         stoi_MF, 
                         itos, 
                         itos_MF):

    mode = "val"
    CLIP_model, val_dataloader = load_data_and_CLIP_model(config, 
                                                                mode, 
                                                                stoi, 
                                                                stoi_MF, 
                                                                itos, 
                                                                itos_MF)
    
    model_MMT = CLIP_model.CLIP_model.MT_model
    model_MMT, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF)    ################# CHANGE WHEN FINAL MMT WITH CLIP IS WORKING
    model_CLIP = CLIP_model.CLIP_model
    
    results_dict = defaultdict(list)
    gen_conv_SMI_list = []
    token_probs_list = []
    src_HSQC_list = []

    # generate all the smiles of trg and greedy gen
    for i, data_dict in tqdm(enumerate(val_dataloader)):
        gen_conv_SMI_list = []
        data_dict_dup = duplicate_dict(data_dict, 1)  # Maybe should hardcode it here as 64 - it will always cut it down to the number needed with ntimes

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                       data_dict_dup, 
                                                                       config)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        greedy_tensor, greedy_token_prob = vgmmt.greedy_sequence(model_MMT, stoi, itos, memory, src_padding_mask, config)
        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor.squeeze(1), greedy_token_prob, itos)
        token_probs_list.append(token_probs)
        gen_conv_SMI_list.append(gen_conv_SMI)   
        src_HSQC_list.append(src_HSQC)
    

        gen_conv_SMI_list = gen_conv_SMI_list[:1]

        mean_loss, losses, logits, targets, dot_similarity= model_CLIP.inference(data_dict_dup, gen_conv_SMI_list)

        combined_list = [[smile, num.item(), cosim.item()] for smile, num, cosim in zip(gen_conv_SMI_list, losses, dot_similarity)]
        ### Sort by the lowest similarity
        #sorted_list = sorted(combined_list, key=lambda x: x[1])
        
        combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
        combined_list, batch_data = add_HSQC_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.MMT_batch)

        sorted_list = sorted(combined_list, key=lambda x: -x[3]) # SMILES = 0, losses =1, dot_sim= 2, tanimoto = 3 

        results_dict[trg_conv_SMI] = [sorted_list, batch_data]
    #if i == config.n_samples:
    #    break

    return results_dict

def calculate_HSQC_COSY_error(config, data_dict_dup, gen_conv_SMI):
    succ_gen_list = []
    
    src_HSQC_list = [data_dict_dup["src_HSQC"]]
    src_COSY_list = [data_dict_dup["src_COSY"]]
    tensor_HSQC = prepare_HSQC_data_from_src(src_HSQC_list)
    tensor_COSY = prepare_COSY_data_from_src(src_COSY_list)
    for i in range(len(gen_conv_SMI)):
        ran_num = random.randint(0, 1000000)
        sample_id = "0_"+str(ran_num)
        succ_gen_list.append([sample_id, gen_conv_SMI[i]])
    df_succ_smis = pd.DataFrame(succ_gen_list, columns=['sample-id', 'SMILES'])

    avg_sim_error_HSQC, avg_sim_error_COSY, sim_error_list_HSQC, sim_error_list_COSY, batch_data = ttf.run_sgnn_sim_calculations_if_possible_return_spectra(df_succ_smis, tensor_HSQC, tensor_COSY, sgnn_means_stds, config)
    return avg_sim_error_HSQC, avg_sim_error_COSY, sim_error_list_HSQC, sim_error_list_COSY, batch_data


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

def run_test_performance_CLIP_greedy2(config, 
                         stoi, 
                         stoi_MF, 
                         itos, 
                         itos_MF):

    mode = "val"
    CLIP_model, val_dataloader = load_data_and_CLIP_model(config, 
                                                                mode, 
                                                                stoi, 
                                                                stoi_MF, 
                                                                itos, 
                                                                itos_MF)
    
    model_MMT = CLIP_model.CLIP_model.MT_model
    model_MMT, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF)    ################# CHANGE WHEN FINAL MMT WITH CLIP IS WORKING
    model_CLIP = CLIP_model.CLIP_model
    
    results_dict = defaultdict(list)
    gen_conv_SMI_list = []
    token_probs_list = []
    src_HSQC_list = []
    failed = []

    # generate all the smiles of trg and greedy gen
    for i, data_dict in tqdm(enumerate(val_dataloader)):

        data_dict_dup = duplicate_dict(data_dict, 1)  # Maybe should hardcode it here as 64 - it will always cut it down to the number needed with ntimes

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = run_model(model_MMT,
                                                                        data_dict_dup, 
                                                                        config)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        greedy_tensor, greedy_token_prob = vgmmt.greedy_sequence(model_MMT, stoi, itos, memory, src_padding_mask, config)
        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor.squeeze(1), greedy_token_prob, itos)
        tan_sim = try_calculate_tanimoto_from_two_smiles(trg_conv_SMI, gen_conv_SMI, 512, extra_info = False)
        mean_loss, losses, logits, targets, dot_similarity = model_CLIP.inference(data_dict_dup, gen_conv_SMI)

        avg_sim_error_HSQC, avg_sim_error_COSY, sim_error_list_HSQC, sim_error_list_COSY, batch_data = calculate_HSQC_COSY_error(config, data_dict_dup, gen_conv_SMI)
        combined_list =[gen_conv_SMI, losses.item(), dot_similarity.item(), tan_sim, [avg_sim_error_HSQC, avg_sim_error_COSY]]
        if tan_sim == None:
            failed.append([trg_conv_SMI, gen_conv_SMI, combined_list, batch_data])
            continue
        else:
            results_dict[trg_conv_SMI] = [combined_list, batch_data]
    #if i == config.n_samples:
    #    break

    return results_dict, failed




def run_test_performance_CLIP_2(config, 
                         stoi, 
                         stoi_MF, 
                         itos, 
                         itos_MF):

    mode = "val"
    # CLIP_model, val_dataloader = load_data_and_CLIP_model(config, 
    #                                                             mode, 
    #                                                             stoi, 
    #                                                             stoi_MF, 
    #                                                             itos, 
    #                                                             itos_MF)
    # model_CLIP = CLIP_model.CLIP_model

    # model_MMT = CLIP_model.CLIP_model.MT_model
    model_MMT, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF)    ################# CHANGE WHEN FINAL MMT WITH CLIP IS WORKING
    
    total_results = {}
    # Number of times to duplicate each tensor 
    gen_num = 1 # Multinomial sampling number
    prob_dict_results, sample_dict = vgmmt.calculate_corr_max_multinom_prob(model_MMT, stoi, val_dataloader, gen_num, config)
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
    total_results["statistics_multiplication_avg"] = [final_prob_corr_multi_avg,
                                                    final_prob_max_multi_avg,
                                                    final_prob_multinom_multi_avg]

    total_results["statistics_multiplication_sum"] = [final_prob_corr_multi_sum,
                                                    final_prob_max_multi_sum,
                                                    final_prob_multinom_multi_sum]

    total_results["statistics_avg_avg"] = [final_prob_corr_avg_avg,
                                            final_prob_max_avg_avg,
                                            final_prob_multinom_avg_avg, ]
    return total_results


# def run_test_performance_MMT(config, 
#                          stoi, 
#                          stoi_MF, 
#                          itos, 
#                          itos_MF):


#     model_MMT, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF)
    
#     # Number of times to duplicate each tensor
#     n_times = config.multinom_runs

#     results_dict = defaultdict(list)
#     # generate all the smiles of trg and greedy gen
#     for i, data_dict in tqdm(enumerate(val_dataloader)):
#         gen_conv_SMI_list = []

#         trg_enc_SMI = data_dict["trg_enc_SMI"][0]
#         trg_conv_SMI = ttf.tensor_to_smiles(trg_enc_SMI[1:], itos)

#         ### multiply the input to paralellize the generation
#         data_dict_dup = duplicate_dict(data_dict, n_times)

#         memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = run_model(model_MMT,
#                                                                        data_dict_dup, 
#                                                                        config)
#         counter = 0
#         ### Here I increase the temperature in case if it does not generate enough diverse molecules but sets it back to the original
#         ### value after enough molecules were found.
#         temp_orig = config.temperature
#         while len(gen_conv_SMI_list)<n_times:
#             # increase the temperature if not enough different molecules get generated
#             if counter%10==0:
#                 config.temperature = config.temperature + 0.1
#             multinom_tensor, multinom_token_prob = multinomial_sequence(model_MMT, memory, src_padding_mask, stoi, config)
#             gen_conv_SMI, token_probs = ttf.tensor_to_smiles_and_prob(multinom_tensor, multinom_token_prob, itos)
#             gen_conv_SMI = filter_valid_smiles_and_canonicolize(gen_conv_SMI)
#             gen_conv_SMI_list.extend(gen_conv_SMI)
#             gen_conv_SMI_list = list(set(gen_conv_SMI_list))
#             counter += 1
#         print(config.temperature)
#         config.temperature = temp_orig

#         gen_conv_SMI_list = gen_conv_SMI_list[:n_times]

#         #mean_loss, losses, logits, targets, dot_similarity= model_CLIP.inference(data_dict_dup,gen_conv_SMI_list)

# #        combined_list = [[smile, num.item(), cosim.item()] for smile, num, cosim in zip(gen_conv_SMI_list, losses, dot_similarity)]
#         combined_list = [[smile, 0, 0] for smile in gen_conv_SMI_list]  # because no CLIP available for this one yet
#         ### Sort by the lowest similarity
#         #sorted_list = sorted(combined_list, key=lambda x: x[1])

#         combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
#         combined_list, batch_data = add_HSQC_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.MMT_batch)

#         sorted_list = sorted(combined_list, key=lambda x: -x[3]) # SMILES = 0, losses =1, dot_sim= 2, tanimoto = 3 

#         results_dict[trg_conv_SMI] = [sorted_list, batch_data]
#         if i == config.n_samples:
#             break

#     return results_dict
    


from io import BytesIO
import base64
def plot_hist_of_results(results_dict):
    # Initialize an empty list to store the 3rd elements
    tani_list = []

    # Loop over each key in results_dict
    for key in tqdm(results_dict.keys()):
        #try:
            sorted_list = results_dict[key]  # The sorted_list is the first element of the value list

            # Check if sorted_list has enough elements
            third_element = sorted_list[0][0][4]  # Take the 3rd element Tanimoto similarity
            tani_list.append(third_element)


    fig, ax = plt.subplots()

    # Create histogram
    ax.hist(tani_list, bins=20, edgecolor='black')

    # Add title and labels
    plt.title(f'Histogram of best Tanimoto Similarity: Sample size {len(results_dict)}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()
        # Save the plot to a BytesIO object and encode it as base64
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)  # Close the figure to free memory

    if len(tani_list) != 0:
        avg_tani = sum(tani_list) / len(tani_list)
    else:
        avg_tani = 0
    return avg_tani, html



def plot_hist_of_results_greedy(results_dict):
    # Initialize an empty list to store the 3rd elements
    tani_list = []

    # Loop over each key in results_dict
    for key in tqdm(results_dict.keys()):
        #try:
            sorted_list = results_dict[key]  # The sorted_list is the first element of the value list
            third_element = sorted_list[0][3]  # Take the 3rd element Tanimoto similarity
            tani_list.append(third_element)


    fig, ax = plt.subplots()

    # Create histogram
    ax.hist(tani_list, bins=20, edgecolor='black')

    # Add title and labels
    plt.title(f'Histogram of best Tanimoto Similarity: Sample size {len(results_dict)}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()
        # Save the plot to a BytesIO object and encode it as base64
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)  # Close the figure to free memory


    avg_tani = sum(tani_list) / len(tani_list)
    return avg_tani, html


def plot_hist_of_results_greedy_new(results_dict):
    # Extract the Tanimoto similarities
    tani_list = results_dict.get('tanimoto_sim', [])
    
    # Filter out zeros
    tani_list = [t for t in tani_list if t != 0]
    
    # Count successful and failed generations
    successful = len(tani_list)
    failed = len(results_dict.get('tanimoto_sim', [])) - successful

    fig, ax = plt.subplots()

    # Create histogram
    ax.hist(tani_list, bins=20, edgecolor='black')

    # Add title and labels
    plt.title(f'Histogram of Greedy Sampled Tanimoto Similarity: {successful} Successful, {failed} Failed Molecules')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

    avg_tani = sum(tani_list) / len(tani_list) if tani_list else 0
    return avg_tani