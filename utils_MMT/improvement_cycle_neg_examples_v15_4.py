import os
from typing import List, Dict, Any, Union, Tuple
import pandas as pd
from datetime import datetime
import pickle
import re
import tempfile
import os
import sys
import copy
import pickle
import tempfile
import re
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime
import json
import random

# Data processing
import pandas as pd
import numpy as np

# Custom modules (based on function calls in the code)
import utils_MMT.helper_functions_pl_v15_4 as hf
import utils_MMT.clustering_visualization_v15_4 as cv
import utils_MMT.execution_function_v15_4 as ex
import utils_MMT.mmt_result_test_functions_15_4 as mrtf
import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm



def load_json_dics():
    with open('./itos.json', 'r') as f:
        itos = json.load(f)
    with open('./stoi.json', 'r') as f:
        stoi = json.load(f)
    with open('./stoi_MF.json', 'r') as f:
        stoi_MF = json.load(f)
    with open('./itos_MF.json', 'r') as f:
        itos_MF = json.load(f)    
    return itos, stoi, stoi_MF, itos_MF
    
itos, stoi, stoi_MF, itos_MF = load_json_dics()
rand_num = str(random.randint(1, 10000000))


def split_dataset(config, chunk_size: int) -> List[pd.DataFrame]:
    df = pd.read_csv(config.SGNN_csv_gen_smi)
    return [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]


def create_chunk_folder(config, idx: int) -> str:
    base_dir = config.model_save_dir
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_folder_name = f"chunk_{idx:03d}_{current_datetime}"
    chunk_folder_path = os.path.join(base_dir, chunk_folder_name)
    
    os.makedirs(chunk_folder_path, exist_ok=True)
    print(f"Created folder for chunk {idx}: {chunk_folder_path}")
    
    return chunk_folder_path

def test_pretrained_model_on_sim_data_before(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, idx):
    MW_filter, greedy_full = True, False
    
    print("prepare_data")
    config = prepare_data(config, chunk)
    print("generate_simulated_data")
    config = generate_simulated_data(config, IR_config)

    print("load_model_and_data")
    model_MMT, val_dataloader, val_dataloader_multi = load_model_and_data(config, stoi, stoi_MF)

    print("run_model_analysis")
    prob_dict_results_1c_, results_dict_1c_ = mrtf.run_model_analysis(config, model_MMT, val_dataloader_multi, stoi, itos)

    results = test_model_performance(config, model_MMT, val_dataloader, val_dataloader_multi, stoi, itos, stoi_MF, itos_MF)

    save_results_before(results, config, idx)

    return config

def prepare_data(config: Any, chunk: pd.DataFrame) -> Any:
    chunk_csv_path = os.path.join(config.pkl_save_folder, "SGNN_csv_gen_smi.csv")
    chunk.to_csv(chunk_csv_path)
    config.SGNN_csv_gen_smi = chunk_csv_path 
    config.data_size = len(chunk)
    return config

def generate_simulated_data(config: Any, IR_config: Any) -> Any:
    config.execution_type = "data_generation"
    if config.execution_type == "data_generation":
        print("\033[1m\033[31mThis is: data_generation\033[0m")
        #import IPython; IPython.embed();

        config = ex.gen_sim_aug_data(config, IR_config)
        backup_config_paths(config)
    return config

def backup_config_paths(config: Any) -> None:
    config.csv_1H_path_SGNN_backup = copy.deepcopy(config.csv_1H_path_SGNN)
    config.csv_13C_path_SGNN_backup = copy.deepcopy(config.csv_13C_path_SGNN)
    config.csv_HSQC_path_SGNN_backup = copy.deepcopy(config.csv_HSQC_path_SGNN)
    config.csv_COSY_path_SGNN_backup = copy.deepcopy(config.csv_COSY_path_SGNN)
    config.IR_data_folder_backup = copy.deepcopy(config.IR_data_folder)

def save_results_before(results: Dict[str, Any], config: Any, idx: int) -> None:
    variables_to_save = {
        'avg_tani_bl_ZINC': results['avg_tani_bl_ZINC_'],
        'results_dict_greedy_bl_ZINC': results.get('results_dict_greedy_bl_ZINC_'),
        'failed_bl_ZINC': results.get('failed_bl_ZINC_'),
        'avg_tani_greedy_bl_ZINC': results['avg_tani_greedy_bl_ZINC_'],
        'results_dict_ZINC_greedy_bl': results.get('results_dict_ZINC_greedy_bl_'),
        'total_results_bl_ZINC': results['total_results_bl_ZINC_'],
        'corr_sampleing_prob_bl_ZINC': results['corr_sampleing_prob_bl_ZINC_'],
        'results_dict_bl_ZINC': results['results_dict_bl_ZINC_'],
    }
    save_data_with_datetime_index(variables_to_save, config.pkl_save_folder, "before_sim_data", idx)

def create_run_folder(chunk_folder, idx):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{idx}_{current_datetime}"
    run_folder_path = os.path.join(chunk_folder, run_folder_name)
    
    os.makedirs(run_folder_path, exist_ok=True)
    print(f"Created folder for run {idx}: {run_folder_path}")
    
    return run_folder_path

def fine_tune_model_aug_mol(config, IR_config, stoi, stoi_MF, chunk, idx):
    #import IPython; IPython.embed();
    config, all_gen_smis, aug_mol_df = generate_augmented_molecules_from_aug_mol(config, IR_config, chunk, idx)
    
    config.parent_model_save_dir = config.model_save_dir
    config.model_save_dir = config.current_run_folder 
    
    if config.execution_type == "transformer_improvement":
        print("\033[1m\033[31mThis is: transformer_improvement, sim_data_gen == TRUE\033[0m")
        config.training_setup = "pretraining"
        mtf.run_MMT(config, stoi, stoi_MF)
    
    config.model_save_dir = config.parent_model_save_dir
    #config = ex.update_model_path(config)

    return config, aug_mol_df, all_gen_smis


def generate_augmented_molecules_from_aug_mol(config, IR_config, chunk, idx):
    #import IPython; IPython.embed();

    ############# THis is just relevant for the augmented molecules #############
    #chunk.rename(columns={'SMILES': 'SMILES_orig', 'SMILES_regio_isomers': 'SMILES'}, inplace=True)
    #############################################################################
    
    script_dir = os.getcwd()
    
    base_path = os.path.abspath(os.path.join(script_dir, 'deep-molecular-optimization'))

    csv_file_path = f'{base_path}/data/MMP/test_selection_2.csv'
    chunk.to_csv(csv_file_path, index=False)
    print(f"CSV file '{csv_file_path}' created successfully.")

    config.data_size = len(chunk)
    config.n_samples = config.data_size

    config, results_dict_MF = generate_smiles_mf(config)

    combined_list_MF = process_generated_smiles(results_dict_MF, config)

    all_gen_smis = filter_and_combine_smiles(config, combined_list_MF)

    aug_mol_df = create_augmented_dataframe(all_gen_smis)

    config, final_df = ex.blend_aug_with_train_data(config, aug_mol_df)

    config = ex.gen_sim_aug_data(config, IR_config)
    config.execution_type = "transformer_improvement"

    return config, all_gen_smis, aug_mol_df


def fine_tune_model(config, stoi, stoi_MF, chunk, idx):
    """
    Fine-tune the model on a chunk of data.
    """
    config, aug_mol_df, all_gen_smis = generate_augmented_molecules(config, chunk, idx)
    
    config.parent_model_save_dir = config.model_save_dir
    new_model_save_dir = create_model_save_dir(config.parent_model_save_dir, idx)
    config.model_save_dir = new_model_save_dir
    
    # Fine-tune the model
    if config.execution_type == "transformer_improvement":
        print("\033[1m\033[31mThis is: transformer_improvement, sim_data_gen == TRUE\033[0m")
        config.training_setup = "pretraining"
        mtf.run_MMT(config, stoi, stoi_MF)
        
    #config = ex.update_model_path(config)
    config.model_save_dir = config.parent_model_save_dir
    
    return config, aug_mol_df, all_gen_smis

def generate_augmented_molecules(config, chunk, idx):
    #import IPython; IPython.embed();
    script_dir = os.getcwd()
    
    base_path = os.path.abspath(os.path.join(script_dir, 'deep-molecular-optimization'))

    csv_file_path = f'{base_path}/data/MMP/test_selection_2.csv'
    chunk.to_csv(csv_file_path, index=False)
    print(f"CSV file '{csv_file_path}' created successfully.")

    config.data_size = len(chunk)
    config.n_samples = config.data_size

    config, results_dict_MF = generate_smiles_mf(config)

    combined_list_MF = process_generated_smiles(results_dict_MF, config)

    all_gen_smis = filter_and_combine_smiles(config, combined_list_MF)

    aug_mol_df = create_augmented_dataframe(all_gen_smis)

    config, final_df = ex.blend_aug_with_train_data(config, aug_mol_df)

    config = ex.gen_sim_aug_data(config, IR_config)
    config.execution_type = "transformer_improvement"

    return config, all_gen_smis, aug_mol_df


def generate_smiles_mf(config):
    print("\033[1m\033[31mThis is: SMI_generation_MF\033[0m")
    return ex.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)

def process_generated_smiles(results_dict_MF, config):
    results_dict_MF = {key: value for key, value in results_dict_MF.items() if not hf.contains_only_nan(value)}
    for key, value in results_dict_MF.items():
        results_dict_MF[key] = hf.remove_nan_from_list(value)

    combined_list_MF, _, _, _ = cv.plot_cluster_MF(results_dict_MF, config)
    return combined_list_MF

def filter_and_combine_smiles(config, combined_list_MF):
    print("\033[1m\033[31mThis is: combine_MMT_MF\033[0m")
    all_gen_smis = combined_list_MF
    all_gen_smis = [smiles for smiles in all_gen_smis if smiles != 'NAN']

    val_data = pd.read_csv(config.csv_path_val)
    all_gen_smis = mrtf.filter_smiles(val_data, all_gen_smis)
    return all_gen_smis

def create_augmented_dataframe(all_gen_smis):
    length_of_list = len(all_gen_smis)
    random_number_strings = [f"GT_{str(i).zfill(7)}" for i in range(1, length_of_list + 1)]
    return pd.DataFrame({'SMILES': all_gen_smis, 'sample-id': random_number_strings})

def setup_data_paths(config):
    base_path_acd = "/projects/cc/se_users/knlr326/1_NMR_project/1_NMR_data_AZ/37_Richard_ACD_sim_data/"
    config.csv_1H_path_ACD = f"{base_path_acd}ACD_1H_with_SN_filtered_v3.csv"
    config.csv_13C_path_ACD = f"{base_path_acd}ACD_13C_with_SN_filtered_v3.csv"
    config.csv_HSQC_path_ACD = f"{base_path_acd}ACD_HSQC_with_SN_filtered_v3.csv"
    config.csv_COSY_path_ACD = f"{base_path_acd}ACD_COSY_with_SN_filtered_v3.csv"
    config.IR_data_folder_ACD = f"{base_path_acd}IR_spectra"
    
    base_path_exp = "/projects/cc/se_users/knlr326/1_NMR_project/1_NMR_data_AZ/36_Richard_43_dataset/experimenal_data/"
    config.csv_1H_path_exp = f"{base_path_exp}real_1H_with_AZ_SMILES_v3.csv"
    config.csv_13C_path_exp = f"{base_path_exp}real_13C_with_AZ_SMILES_v3.csv"
    config.csv_HSQC_path_exp = f"{base_path_exp}real_HSQC_with_AZ_SMILES_v3.csv"
    config.csv_COSY_path_exp = f"{base_path_exp}real_COSY_with_AZ_SMILES_v3.csv"
    config.IR_data_folder_exp = f"{base_path_exp}IR_data"
    return config

def test_model_on_neg_dataset(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, composite_idx, aug_mol_df, all_gen_smis):
    checkpoint_path_backup = config.checkpoint_path    
    config.pickle_file_path = ""
    config.training_mode = "1H_13C_HSQC_COSY_IR_MF_MW"
    config = test_on_data(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, composite_idx, "sim", aug_mol_df, all_gen_smis)
    config.checkpoint_path = checkpoint_path_backup
    return config

def test_on_data(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, composite_idx, data_type, aug_mol_df, all_gen_smis):
    if data_type == 'sim':
        restore_backup_configs(config)
    #else:
    #    sample_ids = chunk['sample-id'].tolist()
    #    process_spectrum_data(config, sample_ids, data_type)
    #import IPython; IPython.embed();

    update_config_settings(config)
    last_checkpoint = get_last_checkpoint(config.current_run_folder)
    config.checkpoint_path = last_checkpoint
    
    model_MMT, val_dataloader, val_dataloader_multi = load_model_and_data(config, stoi, stoi_MF)
    
    prob_dict_results_1c_, results_dict_1c_ = mrtf.run_model_analysis(config, model_MMT, val_dataloader_multi, stoi, itos)

    results = test_model_performance(config, model_MMT, val_dataloader, val_dataloader_multi,
                                     stoi, itos, stoi_MF, itos_MF)
    
    if data_type == 'sim':
        results['aug_mol_df'] = aug_mol_df
        results['all_gen_smis'] = all_gen_smis
    
    save_results_acd_exp(results, config, data_type, composite_idx)
    return config

def restore_backup_configs(config):
    config.csv_1H_path_SGNN = config.csv_1H_path_SGNN_backup
    config.csv_13C_path_SGNN = config.csv_13C_path_SGNN_backup
    config.csv_HSQC_path_SGNN = config.csv_HSQC_path_SGNN_backup
    config.csv_COSY_path_SGNN = config.csv_COSY_path_SGNN_backup
    config.IR_data_folder = config.IR_data_folder_backup 
    config.csv_path_val = config.csv_1H_path_SGNN_backup
    config.pickle_file_path = ""
    
"""
def process_spectrum_data(config: Any, sample_ids: List[str], data_type: str) -> None:
    spectrum_types = ['1H', '13C', 'HSQC', 'COSY']
    for spectrum in spectrum_types:
        csv_path = getattr(config, f'csv_{spectrum}_path_{data_type}')
        df_data = pd.read_csv(csv_path)
        df_data['sample-id'] = df_data['AZ_Number']
        data = select_relevant_samples(df_data, sample_ids)
        dummy_path, config = save_and_update_config(config, data_type, spectrum, data)
        print(f"Saved {spectrum} data to: {dummy_path}")
    if data_type == "ACD" or data_type == "sim":
        config.IR_data_folder = config.IR_data_folder_backup 
    elif  data_type == "exp":
        config.IR_data_folder = config.IR_data_folder_exp 
"""
    
    
def select_relevant_samples(df: pd.DataFrame, sample_ids: List[str]) -> pd.DataFrame:
    return df[df['sample-id'].isin(sample_ids)]

def save_and_update_config(config, data_type: str, spectrum_type: str, data: pd.DataFrame) -> Tuple[str, Any]:
    temp_dir = tempfile.mkdtemp()
    dummy_path = os.path.join(temp_dir, f"{data_type}_{spectrum_type}_selected_samples.csv")
    
    data.to_csv(dummy_path, index=False)
    
    config_key = f'csv_{spectrum_type}_path_SGNN'
    setattr(config, config_key, dummy_path)
    
    return dummy_path, config

def update_config_settings(config: Any) -> None:
    config.csv_path_val = config.csv_1H_path_SGNN
    config.pickle_file_path = ""

def get_last_checkpoint(model_folder: str) -> str:
    checkpoints = [f for f in os.listdir(model_folder) if f.endswith('.ckpt')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_folder}")
    
    last_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(model_folder, x)))
    return os.path.join(model_folder, last_checkpoint)

def load_model_and_data(config: Any, stoi: Dict, stoi_MF: Dict) -> Tuple[Any, Any, Any]:
    #import IPython; IPython.embed();

    val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    model_MMT = mrtf.load_MMT_model(config)
    return model_MMT, val_dataloader, val_dataloader_multi

def test_model_performance(config: Any, model_MMT: Any, val_dataloader: Any, val_dataloader_multi: Any, 
                           stoi: Dict, itos: Dict, stoi_MF: Dict, itos_MF: Dict) -> Dict[str, Any]:
    print("\033[1m\033[31mThis is: test_performance\033[0m")
    
    MW_filter = True
    greedy_full = False
    
    model_CLIP = mrtf.load_CLIP_model(config)
    
    results = {}
    
    results['results_dict_bl_ZINC_'] = mrtf.run_test_mns_performance_CLIP_3(
        config, model_MMT, model_CLIP, val_dataloader, stoi, itos, MW_filter)
    results['results_dict_bl_ZINC_'], counter = mrtf.filter_invalid_inputs(results['results_dict_bl_ZINC_'])

    results['avg_tani_bl_ZINC_'], html_plot = rbgvm.plot_hist_of_results(results['results_dict_bl_ZINC_'])

    if greedy_full:
        results['results_dict_greedy_bl_ZINC_'], results['failed_bl_ZINC_'] = mrtf.run_test_performance_CLIP_greedy_3(
            config, stoi, stoi_MF, itos, itos_MF)
        results['avg_tani_greedy_bl_ZINC_'], html_plot_greedy = rbgvm.plot_hist_of_results_greedy(
            results['results_dict_greedy_bl_ZINC_'])
    else:
        config, results['results_dict_ZINC_greedy_bl_'] = mrtf.run_greedy_sampling(
            config, model_MMT, val_dataloader_multi, itos, stoi)
        results['avg_tani_greedy_bl_ZINC_'] = results['results_dict_ZINC_greedy_bl_']["tanimoto_mean"]

    results['total_results_bl_ZINC_'] = mrtf.run_test_performance_CLIP_3(
        config, model_MMT, val_dataloader, stoi)
    results['corr_sampleing_prob_bl_ZINC_'] = results['total_results_bl_ZINC_']["statistics_multiplication_avg"][0]

    print("avg_tani, avg_tani_greedy, corr_sampleing_prob'")
    print(results['avg_tani_bl_ZINC_'], results['avg_tani_greedy_bl_ZINC_'], results['corr_sampleing_prob_bl_ZINC_'])
    print("Greedy tanimoto results")
    rbgvm.plot_hist_of_results_greedy_new(results['results_dict_ZINC_greedy_bl_'])

    return results

def save_results_acd_exp(results: Dict[str, Any], config: Any, data_type: str, composite_idx: str) -> None:
    variables_to_save = {
        'avg_tani_bl_ZINC': results['avg_tani_bl_ZINC_'],
        'results_dict_greedy_bl_ZINC': results.get('results_dict_greedy_bl_ZINC_'),
        'failed_bl_ZINC': results.get('failed_bl_ZINC_'),
        'avg_tani_greedy_bl_ZINC': results['avg_tani_greedy_bl_ZINC_'],
        'results_dict_ZINC_greedy_bl': results.get('results_dict_ZINC_greedy_bl_'),
        'total_results_bl_ZINC': results['total_results_bl_ZINC_'],
        'corr_sampleing_prob_bl_ZINC': results['corr_sampleing_prob_bl_ZINC_'],
        'results_dict_bl_ZINC': results['results_dict_bl_ZINC_'],
        'checkpoint_path': config.checkpoint_path,
    }
    
    if data_type == 'sim':
        variables_to_save['aug_mol_df'] = results.get('aug_mol_df')
        variables_to_save['all_gen_smis'] = results.get('all_gen_smis')
    
    save_data_with_datetime_index(
        variables_to_save, 
        config.pkl_save_folder, 
        f"{data_type}_sim_data", 
        composite_idx
    )

def save_data_with_datetime_index(data: Any, base_folder: str, name: str, idx: Union[int, str]) -> None:
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_datetime}_{name}_{idx}.pkl"
    os.makedirs(base_folder, exist_ok=True)
    file_path = os.path.join(base_folder, filename)

    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data saved to: {file_path}")