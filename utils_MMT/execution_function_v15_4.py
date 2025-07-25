import utils_MMT.clip_functions_v15_4 as cf
import utils_MMT.dataloaders_pl_v15_4 as dl
from utils_MMT.nmr_calculation_from_dft_v15_4 import *
import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.data_generation_v15_4 as dg
import utils_MMT.molformer_functions_v15_4 as mff
import utils_MMT.validate_generate_MMT_v15_4 as vgmmt
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
import utils_MMT.clustering_visualization_v15_4 as cv
import utils_MMT.ir_simulation_v15_4 as irs
import utils_MMT.plotting_v15_4 as pt
import utils_MMT.train_test_functions_pl_v15_4 as ttf

from sklearn.utils import shuffle
from argparse import Namespace
import random
import glob


def parse_arguments(hyperparameters):
    # Using dictionary comprehension to simplify your code
    parsed_args = {key: val[0] for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)

def simulate_syn_data(config, IR_config):
    if os.path.exists(config.SGNN_gen_folder_path):
        ran_num = config.ran_num
        print(ran_num)
        config.SGNN_ran_num = ran_num
        path_gen_folder = config.SGNN_gen_folder_path+ "_syn_" + ran_num
        os.mkdir(path_gen_folder)
        config.SGNN_gen_folder_path = path_gen_folder
    config.SGNN_csv_gen_smi = config.csv_SMI_targets
    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)

    #args = parse_arguments(IR_config_dict)
    config.csv_SMI_targets = csv_1H_path # it should just generate the smiles that has been successfully generated for 1H 
    IR_output_path = irs.run_IR_simulation(config, IR_config, "target")

    config.csv_1H_path_SGNN = csv_1H_path
    config.csv_13C_path_SGNN = csv_13C_path 
    config.csv_HSQC_path_SGNN = csv_HSQC_path
    config.csv_COSY_path_SGNN = csv_COSY_path 
    config.IR_data_folder = IR_output_path
    #config.IR_data_folder = ""     ############
    config.pickle_file_path = ""

    config.csv_1H_path_GT_sim = csv_1H_path
    config.csv_13C_path_GT_sim = csv_13C_path 
    config.csv_HSQC_path_GT_sim = csv_HSQC_path
    config.csv_COSY_path_GT_sim = csv_COSY_path
    config.IR_data_folder_GT_sim = IR_output_path
    #config.IR_data_folder_GT_sim = ""  ############
    
    # in case if I get IR as  well
    config.data_size = len(data_1H)
    
    # to make sure that failed molecules get ignored and just the successful
    # generations are processed
    config.csv_path_val = csv_1H_path
    
    #sort automatic model saving in new folder
    config.training_mode = "1H_13C_HSQC_COSY_IR_MF_MW"
    return config


    
def SMI_generation_MMT(config, stoi, stoi_MF, itos, itos_MF):
    #config.data_size = 10 # number of molecules that should be taken for generation of new molecules
    mode = "val"
    sort_num = 3 # 3=tanimoto score
    CLIP_model, dataloader = rbgvm.load_data_and_CLIP_model(config, mode, stoi, stoi_MF, itos, itos_MF)
    MMT_m = CLIP_model.CLIP_model.MT_model
    CLIP_m = CLIP_model.CLIP_model
    MMT_m, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF)    ######################## as long as the CLIP model is not working correctly

    results_dict = rbgvm.run_MMT_generations(config, MMT_m, CLIP_m, dataloader, stoi, itos, sort_num)

    return config, results_dict




def SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF): 

    #mode = "val"
    #df = mff.load_data(config, mode, stoi, stoi_MF, itos, itos_MF)
    #import IPython; IPython.embed();
    combined_df = mff.main_molformer(config)
    #print("combined_df")
    #print(combined_df)
    # Generating the dictionary
    results_dict_MF = {}
    for column in tqdm(combined_df.columns):
        results_dict_MF[combined_df[column][0]] = combined_df[column][1:].tolist()

    # combined_list_MF, html_TSNE, html_UMAP, html_PCA = cv.plot_cluster_MF(results_dict_MF, config, mode, stoi, stoi_MF, itos, itos_MF)
    # #import IPython; IPython.embed();
    # max_num = 20
    # mol_html_plot = pt.plot_molecules_from_list(combined_list_MF, max_num)
    
    return config, results_dict_MF

import copy
def gen_sim_aug_data(config, IR_config):

    ran_num = random.randint(100000, 999999)
    config.ran_num = ran_num
    print("outside ran_num")
    print(ran_num)
    config.SGNN_gen_folder_path_backup = config.SGNN_gen_folder_path

    if os.path.exists(config.SGNN_gen_folder_path):
        print(ran_num)
        path_gen_folder = config.SGNN_gen_folder_path + "_" + str(ran_num)
        os.makedirs(path_gen_folder, exist_ok=True)
        config.SGNN_gen_folder_path = path_gen_folder

    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)

    config.csv_SMI_targets = copy.deepcopy(csv_1H_path) # it should just generate the smiles that has been successfully generated for 1H 
    data_IR = irs.run_IR_simulation(config, IR_config, "target")
    print("\033[1m\033[33m IR Generation: DONE\033[0m")    
    config.csv_1H_path_SGNN = copy.deepcopy(csv_1H_path)
    config.csv_13C_path_SGNN = copy.deepcopy(csv_13C_path) 
    config.csv_HSQC_path_SGNN = copy.deepcopy(csv_HSQC_path)
    config.csv_COSY_path_SGNN = copy.deepcopy(csv_COSY_path) 
    config.IR_data_folder = copy.deepcopy(data_IR)
    #config.IR_data_folder = ""  ### Because for now we don't use it but 15.1 will

    # in case if I get IR as well
    config.data_size = len(data_1H)
    config.pickle_file_path = ""
    # to make sure that failed molecules get ignored and just the successful
    # generations are processed
    config.csv_path_val = csv_1H_path
    config.SGNN_gen_folder_path = config.SGNN_gen_folder_path_backup
    #sort automatic model saving in new folder
    #config.training_mode = "1H_13C_HSQC_COSY_IR_MF_MW"
    #config.execution_type = "transformer_improvement"
    return config


def blend_aug_with_train_data(config, aug_mol_df):
    # Assuming df is already created with 'SMILES' and 'sample-id'
    # df = pd.DataFrame({'SMILES': all_gen_smis, 'sample-id': random_number_strings})
    train_df = pd.read_csv(config.csv_train_path)
    random_subset = train_df.sample(n=config.train_data_blend)
    # Combine the dataframes
    combined_df = pd.concat([random_subset, aug_mol_df], ignore_index=True)
    # Shuffle the combined dataframe
    final_df = shuffle(combined_df)
    # Reset index if needed
    final_df.reset_index(drop=True, inplace=True)
    
    csv_file_path = os.path.join(config.gen_mol_csv_folder_path, f"{config.ran_num}.csv")
    print(csv_file_path)                                                             
    final_df.to_csv(csv_file_path)

    config.SGNN_csv_gen_smi = csv_file_path
    #config.execution_type = "data_generation"
    config.data_size = len(final_df)
    return config, final_df


def update_model_path(config):
    flist = glob.glob(config.model_save_dir+"/*.ckpt")
    #filtered_flist = [file for file in flist if f"epoch={config.num_epochs}" in file]
    flist.sort(key=os.path.getmtime, reverse=True)
    new_model_path = flist[0]
    config.checkpoint_path = new_model_path
    print(config.checkpoint_path)
    return config


# def filter_invalid_criteria(file_path):
#     # Path to the original CSV file
#     #file_path = config.csv_1H_path_SGNN
    
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(file_path)

#     # Filter out rows where the SMILES string contains 'P' or '[NH+]'
#     df_filtered = df[~df['SMILES'].str.contains('P')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains('Si')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\-')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\+')]

#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[B\]')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains('B')]
#     #df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[NH2\+\]')]
#     #df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[c\-\]')]
#     #df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[C\+\]')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[NH\+\]', regex=True)]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[SH\]', regex=True)]

#     def remove_stereo(smiles):
#         smiles = smiles.replace('[C@@H]', 'C').replace('[C@H]', 'C')
#         smiles = smiles.replace('[C@@]', 'C').replace('[C@]', 'C')
#         return smiles

#     # Apply the function to each SMILES string in the DataFrame
#     df_filtered['SMILES'] = df_filtered['SMILES'].apply(remove_stereo)

#     # Generate a random number for the new file name
#     random_number = random.randint(10000, 99999)
    
#     # Construct new file name with the random number
#     new_file_name = file_path.rsplit('.', 1)[0] + f"_{random_number}.csv"

#     # Save the filtered DataFrame to the new file
#     df_filtered.to_csv(new_file_name, index=False)
    
#     # Update the config with the new file path
#     #config.csv_path_val = new_file_name
#     #config.csv_1H_path_SGNN = new_file_name
#     return new_file_name


import pandas as pd
import random
from rdkit import Chem

def filter_invalid_criteria(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Function to canonicalize SMILES strings
    def canonicalize_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None

    # Apply the canonicalization function to each SMILES string in the DataFrame
    df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)

    # Filter out rows where the SMILES string contains specific unwanted elements or characters
    df_filtered = df[~df['SMILES'].str.contains('P')]
    df_filtered = df_filtered[~df['SMILES'].str.contains('Si')]
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\[B\]')]
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\bB\b(?!r)', regex=True)]
    # df_filtered = df_filtered[~df['SMILES'].str.contains('B')]
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\[NH\+\]', regex=True)]
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\[SH\]', regex=True)]

    # Specifically match '-' and '+' as charges, not as part of the structure
    df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[.*?\-.*?\]', regex=True)]
    df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[.*?\+.*?\]', regex=True)]

    def remove_stereo(smiles):
        smiles = smiles.replace('[C@@H]', 'C').replace('[C@H]', 'C')
        smiles = smiles.replace('[C@@]', 'C').replace('[C@]', 'C')
        return smiles

    # Apply the function to remove stereochemistry from each SMILES string in the DataFrame
    df_filtered['SMILES'] = df_filtered['SMILES'].apply(remove_stereo)

    # Drop rows where the canonicalization failed (i.e., resulted in None)
    df_filtered = df_filtered.dropna(subset=['SMILES'])

    # Generate a random number for the new file name
    random_number = random.randint(10000, 99999)
    
    # Construct new file name with the random number
    new_file_name = file_path.rsplit('.', 1)[0] + f"_{random_number}.csv"

    # Save the filtered and canonicalized DataFrame to the new file
    df_filtered.to_csv(new_file_name, index=False)
    
    return new_file_name


def clean_dataset(config):
    # Path to the original CSV file
    print("\033[1m\033[31mThis is: clean_dataset\033[0m")
    file_path = config.csv_SMI_targets

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Function to remove stereochemistry information from a SMILES string
    def remove_stereo(smiles):
        smiles = smiles.replace('[C@@H]', 'C').replace('[C@H]', 'C')
        smiles = smiles.replace('[C@@]', 'C').replace('[C@]', 'C')
        return smiles

    # Apply the function to each SMILES string in the DataFrame
    df['SMILES'] = df['SMILES'].apply(remove_stereo)

    # Remove rows containing 'P' or '+'
    df = df[~df['SMILES'].str.contains('P', regex=True)]
    df = df[~df['SMILES'].str.contains('Si')]

    # Generate a random number for the new file name
    random_number = random.randint(10000, 99999)

    # Construct new file name with the random number
    new_file_name = file_path.rsplit('.', 1)[0] + f"_{random_number}.csv"

    # Save the modified DataFrame to the new file
    df.to_csv(new_file_name, index=False)

    # Update the config with the new file path
    config.csv_SMI_targets = new_file_name
    return config