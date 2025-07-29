"""
MMST script for structure prediction workflow.
This script implements the improvement cycle for molecular structure prediction.
"""

# Standard library imports
import os
import sys
import json
import pickle
import time
import logging
import datetime
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any
import pickle

# Third-party imports
import pandas as pd
import numpy as np
import torch

# Third-party imports
import random
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles
# Import path utilities
import sys
from pathlib import Path
# Get relative path to utils_MMT directory
script_dir = Path(__file__).resolve().parent
mmt_dir = script_dir.parent.parent.parent  # Go up to MMT_explainability directory
utils_mmt_dir = mmt_dir / "utils_MMT"
sys.path.append(str(utils_mmt_dir))


# Add necessary directories to Python path
# Get all relevant paths
script_dir = Path(__file__).resolve().parent  # scripts directory
llm_dir = script_dir.parent.parent  # LLM_Structure_Elucidator directory
mmt_dir = llm_dir.parent  # MMT_explainability directory
mol_opt_dir = mmt_dir / "deep-molecular-optimization"  # deep-molecular-optimization directory

# Clear any existing paths that might conflict
sys.path = [p for p in sys.path if not any(str(d) in p for d in [script_dir, llm_dir, mmt_dir, mol_opt_dir])]

# Add directories to path in the correct order
sys.path.insert(0, str(mol_opt_dir))  # First priority for models.dataset
sys.path.insert(0, str(mmt_dir))      # Second priority for utils_MMT
sys.path.insert(0, str(script_dir))   # Top priority for imports_MMST

print("Python paths added in order:")
print(f"1. Script dir: {script_dir}")
print(f"2. MMT dir: {mmt_dir}")
print(f"3. Mol-opt dir: {mol_opt_dir}")
print("\nFull Python path:")
for p in sys.path[:5]:  # Show first 5 paths
    print(f"  {p}")

# Now import the modules after path setup
from utils_MMT.execution_function_v15_4 import *
from utils_MMT.mmt_result_test_functions_15_4 import *
import utils_MMT.data_generation_v15_4 as dg

# Import the modules
from imports_MMST import (
    # Local utilities
    mtf, ex, mrtf, hf,
    # Helper functions
    parse_arguments, load_config, save_updated_config,
    load_configs, load_json_dics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_molecular_data(json_path: str) -> Dict:
    """Load molecular data from JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def test_pretrained_model(config: Dict, stoi: Dict, itos: Dict, stoi_MF: Dict) -> Tuple[float, Dict, Dict]:
    """Test the pre-trained model performance.
    
    Args:
        config: Configuration dictionary
        stoi: String to index mapping
        itos: Index to string mapping
        stoi_MF: String to index mapping for molecular fingerprints
        
    Returns:
        Tuple containing:
        - Performance score
        - Results dictionary
        - Greedy results dictionary
    """
    logger.info("Testing pre-trained model performance...")
    

    # Read input CSV
    input_df = pd.read_csv(config.input_csv)
    if len(input_df) != 1:
        raise ValueError("Input CSV must contain exactly one sample")
    
    # Verify sample-id column exists
    if 'sample-id' not in input_df.columns:
        raise ValueError("Input CSV must have a 'sample-id' column")
    
    # Create SGNN input directory if it doesn't exist
    sgnn_input_dir = Path(config.sgnn_gen_folder) / "input"
    sgnn_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input sample for SGNN
    sgnn_input_file = sgnn_input_dir / "input_sample.csv"
    input_df.to_csv(sgnn_input_file, index=False)
    config.SGNN_csv_gen_smi = str(sgnn_input_file)

    # Generate SGNN data first
    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)
    
    # Update config with generated paths
    config.csv_1H_path_SGNN = csv_1H_path
    config.csv_13C_path_SGNN = csv_13C_path
    config.csv_COSY_path_SGNN = csv_COSY_path
    config.csv_HSQC_path_SGNN = csv_HSQC_path
    
    # Set pickle file path to empty to force regeneration
    config.pickle_file_path = ""
        
    # Log paths for debugging
    logger.info(f"Generated paths:")
    logger.info(f"1H: {csv_1H_path}")
    logger.info(f"13C: {csv_13C_path}")
    logger.info(f"COSY: {csv_COSY_path}")
    logger.info(f"HSQC: {csv_HSQC_path}")
    
    # Set up validation data paths for SGNN
    config.csv_path_val = csv_1H_path  # Set the validation path to 1H data
    config.ref_data_type = "1H"        # Set reference data type to 1H
    config.dl_mode = "val"             # Set mode to validation
    config.training_mode = "1H_13C_HSQC_COSY_MF_MW"  # Set training modes
    # config.training_mode = "HSQC_MF_MW"  # Set training modes
    config.data_type = "sgnn"          # Ensure data type is set to sgnn
    config.data_size = 1               # Set data size to 1 for single sample validation
    
    # Log config paths after setting
    logger.info(f"Config paths after setting:")
    logger.info(f"1H SGNN: {config.csv_1H_path_SGNN}")
    logger.info(f"13C SGNN: {config.csv_13C_path_SGNN}")
    logger.info(f"COSY SGNN: {config.csv_COSY_path_SGNN}")
    logger.info(f"HSQC SGNN: {config.csv_HSQC_path_SGNN}")
    logger.info(f"Val path: {config.csv_path_val}")
    logger.info(f"Ref data type: {config.ref_data_type}")
    logger.info(f"DL mode: {config.dl_mode}")
    logger.info(f"Training mode: {config.training_mode}")
    
    # Load models
    model_MMT = mrtf.load_MMT_model(config)
    model_CLIP = mrtf.load_CLIP_model(config)

    # Load data with the newly generated NMR  contain exactly one sample"
    val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    
    # Run tests
    results_dict = mrtf.run_test_mns_performance_CLIP_3(
        config, model_MMT, model_CLIP, val_dataloader, stoi, itos, True)
    
    results_dict, counter = mrtf.filter_invalid_inputs(results_dict)
    
    # Run greedy sampling
    config, results_dict_greedy = mrtf.run_greedy_sampling(
        config, model_MMT, val_dataloader_multi, itos, stoi)
    
    # Calculate performance metrics
    total_results = mrtf.run_test_performance_CLIP_3(
        config, model_MMT, val_dataloader, stoi)
    
    performance = total_results["statistics_multiplication_avg"][0]
    
    logger.info(f"Model performance: {performance}")
    return performance, results_dict, results_dict_greedy


def run_experimental_data(config: Dict, stoi: Dict, itos: Dict, stoi_MF: Dict) -> Tuple[float, Dict, Dict]:
    """Run the model on experimental data and generate molecules.
    
    Args:
        config: Configuration dictionary
        stoi: String to index mapping
        itos: Index to string mapping
        stoi_MF: String to index mapping for molecular fingerprints
        
    Returns:
        Tuple containing:
        - Performance score
        - Results dictionary
        - Greedy results dictionary
    """
    logger.info(f"Running model on experimental data with {config.multinom_runs} multinomial sampling runs...")
    
    # Log experimental data paths
    logger.info(f"Experimental data paths:")
    logger.info(f"1H: {config.csv_1H_path_SGNN}")
    logger.info(f"13C: {config.csv_13C_path_SGNN}")
    logger.info(f"COSY: {config.csv_COSY_path_SGNN}")
    logger.info(f"HSQC: {config.csv_HSQC_path_SGNN}")
    
    # Configure for experimental data testing
    config.dl_mode = "val"
    config.training_mode = "1H_13C_HSQC_COSY_MF_MW"
    # config.training_mode = "HSQC_MF_MW"  # Set training modes
    config.data_type = "sgnn"
    config.data_size = 1  # Note: data_size remains at 1 for single sample validation
    
    # Load models
    model_MMT = mrtf.load_MMT_model(config)
    model_CLIP = mrtf.load_CLIP_model(config)
    
    # Load experimental data
    val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    
    # Run tests and generate molecules
    results_dict = mrtf.run_test_mns_performance_CLIP_3(
        config, model_MMT, model_CLIP, val_dataloader, stoi, itos, True)
    
    results_dict, counter = mrtf.filter_invalid_inputs(results_dict)
    
    # Run greedy sampling to generate molecules
    config, results_dict_greedy = mrtf.run_greedy_sampling(
        config, model_MMT, val_dataloader_multi, itos, stoi)
    
    # Calculate performance metrics
    total_results = mrtf.run_test_performance_CLIP_3(
        config, model_MMT, val_dataloader, stoi)
    
    performance = total_results["statistics_multiplication_avg"][0]
    
    logger.info(f"Experimental data performance: {performance}")
    return performance, results_dict, results_dict_greedy


def run_improvement_cycle(config: Dict, stoi: Dict, itos: Dict, stoi_MF: Dict, itos_MF: Dict, IR_config: Dict) -> float:
    """Run the improvement cycle workflow.
    
    Args:
        config: Configuration dictionary
        stoi: String to integer mapping
        itos: Integer to string mapping
        stoi_MF: String to integer mapping for molecular formulas
        itos_MF: Integer to string mapping for molecular formulas
        IR_config: IR configuration dictionary
        
    Returns:
        Final performance score
    """
    logger.info("Starting improvement cycle...")
    
    performance = 0
    iteration = 0  # Initialize iteration counter
    max_attempts = 1  # Maximum number of attempts per improvement cycle
    ic_results = []  # Initialize list to store iteration results
    
    # Store all generated molecules across iterations
    all_generated_molecules = set()
    best_molecules = set()  # Store molecules from the best performing iteration
    best_performance = 0
    
    while True:
        logger.info(f"Starting improvement cycle iteration {iteration}")
        logger.info(f"Before loop: iteration={iteration}, max_attempts={max_attempts}")
        if iteration >= max_attempts:
            logger.info(f"Break condition met: {iteration} >= {max_attempts}")
            logger.warning(f"Maximum attempts ({max_attempts}) reached without meeting performance threshold. Breaking cycle.")
            break
        logger.info(f"After check: loop continuing, iteration={iteration}")
            
        # Step 1: Generate molecules using MF
        # For iterations after the first, use molecules from previous iterations as input
        if iteration > 0 and best_molecules:
            logger.info(f"Using {len(best_molecules)} molecules from previous iterations as starting points")
            # Create a temporary dataframe with the best molecules from previous iterations
            temp_smiles = list(best_molecules)
            temp_ids = [f"PREV_{str(i).zfill(7)}" for i in range(1, len(temp_smiles) + 1)]
            temp_df = pd.DataFrame({'SMILES': temp_smiles, 'sample-id': temp_ids})
            
            # Save this dataframe to a temporary file
            temp_file = Path(config.MF_csv_source_folder_location) / f"temp_iteration_{iteration}.csv"
            temp_df.to_csv(temp_file, index=False)
            
            # Temporarily update config to use this file
            original_source_file = config.MF_csv_source_file_name
            config.MF_csv_source_file_name = f"temp_iteration_{iteration}"
            logger.info(f"Using temp file: {temp_file}")
            # Generate new molecules based on previous best
            config, results_dict_MF = ex.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)
            
            # Restore original config
            config.MF_csv_source_file_name = original_source_file
            
            # Clean up temporary file
            if temp_file.exists():
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {temp_file}: {str(e)}")
        else:
            # First iteration - use original input
            config, results_dict_MF = ex.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)
            
        logger.info(f"Generated molecules using MF: {results_dict_MF}")

        # Clean up results
        results_dict_MF = {key: value for key, value in results_dict_MF.items() 
                          if not hf.contains_only_nan(value)}
        for key, value in results_dict_MF.items():
            results_dict_MF[key] = hf.remove_nan_from_list(value)
        
        # Transform results
        transformed_list_MF = [[key, value] for key, value in results_dict_MF.items()]
        src_smi_MF = list(results_dict_MF.keys())
        combined_list_MF = [item for sublist in transformed_list_MF for item in sublist[1][:]]
        
                # Step 2: Combine results
        all_gen_smis = combined_list_MF
        all_gen_smis = [smiles for smiles in all_gen_smis if smiles != 'NAN']

        # If no molecules were generated, use the source SMILES as a fallback
        if not all_gen_smis and src_smi_MF:
            all_gen_smis = [src_smi_MF]
            logger.info(f"No molecules generated, using source SMILES as fallback: {src_smi_MF}")
        else:
            logger.info(f" src_smi_MF: {src_smi_MF}")
            # Add source SMILES for better out of distribution testing (with smiles augmentation)
            if src_smi_MF and len(src_smi_MF) > 0:
                all_gen_smis.append(src_smi_MF[0])
        
        # Store the original generated SMILES before randomization
        original_molformer_smiles = set(all_gen_smis)
        logger.info(f"Storing {len(original_molformer_smiles)} original MolFormer-generated SMILES")

        # For iterations after the first, add best molecules from previous iterations
        if iteration > 0 and best_molecules:
            logger.info(f"Adding {len(best_molecules)} best molecules from previous iterations")
            all_gen_smis.extend(list(best_molecules))
            # Remove duplicates
            all_gen_smis = list(set(all_gen_smis))
        
        # Filter potential hits
        val_data = pd.read_csv(config.csv_path_val)
        all_gen_smis = mrtf.filter_smiles(val_data, all_gen_smis)
        
        # Generate non-canonical SMILES variations
        logger.info("Generating non-canonical SMILES variations")
        
        # Hardcoded parameters for testing
        num_variations = 200  # Number of variations per molecule

        # Generate variations for all SMILES
        expanded_smiles_list = []
        original_count = len(all_gen_smis)
        
        for i, smi in enumerate(tqdm(all_gen_smis, desc="Generating SMILES variations")):
            try:
                # Generate variations for this SMILES
                mol = Chem.MolFromSmiles(smi)
                variations = [Chem.MolToSmiles(mol, doRandom=True) for _ in range(num_variations)]
                
                # Add all variations to the expanded list
                expanded_smiles_list.extend(variations)
                
                # Add the original SMILES to our tracking set
                all_generated_molecules.add(smi)
            except Exception as e:
                logger.warning(f"Error generating variations for SMILES {i}: {str(e)}")
                # Keep the original SMILES if we can't generate variations
                expanded_smiles_list.append(smi)
                all_generated_molecules.add(smi)
        
        # Replace the original list with the expanded one
        all_gen_smis = expanded_smiles_list
        random.shuffle(all_gen_smis)

        # Log expansion statistics
        logger.info(f"SMILES expansion: {original_count} original → {len(all_gen_smis)} with variations")
        logger.info(f"Average variations per molecule: {len(all_gen_smis) / original_count:.2f}")
        
        # Create DataFrame
        length_of_list = len(all_gen_smis)   
        random_number_strings = [f"GT_{str(i).zfill(7)}" for i in range(1, length_of_list + 1)]
        aug_mol_df = pd.DataFrame({'SMILES': all_gen_smis, 'sample-id': random_number_strings})

        # Step 3: Blend with training data
        config.train_data_blend = 0
        config, final_df = ex.blend_aug_with_train_data(config, aug_mol_df)
        
        # Step 4: Generate data
        config = ex.gen_sim_aug_data(config, IR_config)

        # Step 5: Train transformer
        config.training_setup = "pretraining"
        mtf.run_MMT(config, stoi, stoi_MF)
        
        # Update model path and test configuration
        config = ex.update_model_path(config)
        
        # Test current performance
        performance, results_dict, results_dict_greedy = test_pretrained_model(config, stoi, itos, stoi_MF)
        logger.info(f"results_dict: {results_dict}")
        logger.info(f"Current cycle performance: {performance}")
        
        # Extract molecules from results for next iteration
        current_molecules = set()
        if results_dict:
            # Add the source molecules (keys of the dictionary)
            for source_smiles in results_dict.keys():
                if isinstance(source_smiles, str) and Chem.MolFromSmiles(source_smiles) is not None:
                    current_molecules.add(source_smiles)
            
            # Extract generated molecules from the results
            for source_smiles, generated_data in results_dict.items():
                for molecule_data in generated_data:
                    # The first element of each molecule_data list is the SMILES string
                    if isinstance(molecule_data, list) and len(molecule_data) > 0:
                        generated_smiles = molecule_data[0]
                        if isinstance(generated_smiles, str) and Chem.MolFromSmiles(generated_smiles) is not None:
                            current_molecules.add(generated_smiles)
                            
        # Add the original MolFormer-generated SMILES to the current molecules
        if 'original_molformer_smiles' in locals():
            logger.info(f"Adding {len(original_molformer_smiles)} original MolFormer-generated SMILES to current molecules")
            current_molecules.update(original_molformer_smiles)
            logger.info(f"Current molecules count after adding MolFormer SMILES: {len(current_molecules)}")
        # Update best molecules if this iteration has better performance
        if performance > best_performance:
            best_performance = performance
            best_molecules = current_molecules
            logger.info(f"New best performance: {best_performance} with {len(best_molecules)} molecules")
        
        # Save iteration results
        iteration_results = {
            'iteration': iteration,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance': performance,
            'results': results_dict,
            'greedy_results': results_dict_greedy,
            'molecules_count': len(current_molecules),
            'config': vars(config)
        }
        ic_results.append(iteration_results)
        
        # Save all generated molecules for this iteration
        iteration_molecules_file = Path(config.output_dir) / f"iteration_{iteration}_molecules.json"
        with open(iteration_molecules_file, 'w') as f:
            json.dump({
                'performance': performance,
                'molecules': list(current_molecules),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        logger.info(f"Saved {len(current_molecules)} molecules from iteration {iteration} to {iteration_molecules_file}")
        
        if performance > config.IC_threshold:
            logger.info(f"Performance threshold met: {performance} > {config.IC_threshold}")
            break
            
        iteration += 1
    
    # Save all generated molecules across all iterations
    all_molecules_file = Path(config.output_dir) / "all_generated_molecules.json"
    with open(all_molecules_file, 'w') as f:
        json.dump({
            'count': len(all_generated_molecules),
            'molecules': list(all_generated_molecules),
            'best_performance': best_performance,
            'best_molecules': list(best_molecules),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    logger.info(f"Saved {len(all_generated_molecules)} molecules from all iterations to {all_molecules_file}")
    
    return performance, ic_results

def prepare_experimental_data(nmr_type, json_path, output_csv, input_csv):
    """Prepare experimental NMR data from JSON for model input.
    
    Args:
        nmr_type (str): Type of NMR data ('1H', '13C', 'HSQC', 'COSY')
        json_path (str): Path to JSON file containing molecular data
        output_csv (str): Path to save the output CSV
        input_csv (str): Path to input CSV containing sample ID
    """
    # Read sample ID from input CSV
    input_df = pd.read_csv(input_csv)
    if 'sample-id' not in input_df.columns:
        raise ValueError("Input CSV must contain a 'sample-id' column with the sample ID")
    if len(input_df) != 1:
        raise ValueError("Input CSV must contain exactly one row")
    
    target_sample_id = input_df['sample-id'].iloc[0]
    logger.info(f"Processing sample ID: {target_sample_id}")
    
    # Load experimental data
    with open(json_path, 'r') as f:
        molecular_data = json.load(f)
    
    # Extract data for the specified sample
    if target_sample_id not in molecular_data:
        raise ValueError(f"Sample ID {target_sample_id} not found in molecular data")
    
    data = molecular_data[target_sample_id]
    logger.info(f"Data structure for sample {target_sample_id}: {data}")
    logger.info(f"Keys in data: {list(data.keys())}")
    
    # Access nmr_data through molecule_data
    if 'molecule_data' not in data or 'nmr_data' not in data['molecule_data']:
        raise ValueError(f"No NMR data found for sample {target_sample_id}")
    
    nmr_data = data['molecule_data']['nmr_data']
    logger.info(f"Keys in nmr_data: {list(nmr_data.keys())}")
    if f'{nmr_type}_exp' not in nmr_data:
        raise ValueError(f"No {nmr_type} experimental data found for sample {target_sample_id}")
    
    # Create output data
    exp_data = [{
        'sample-id': target_sample_id,
        'SMILES': data['smiles'],
        'NMR_Data': nmr_data[f'{nmr_type}_exp']
    }]
    
    # Save to CSV in required format
    df = pd.DataFrame(exp_data)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved experimental data for sample {target_sample_id} to {output_csv}")
    return output_csv

def get_sample_id_from_csv(csv_path: str) -> str:
    """Extract sample ID from input CSV file.
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        Sample ID string
    """
    try:
        df = pd.read_csv(csv_path)
        if 'sample-id' in df.columns:
            return str(df['sample-id'].iloc[0])
        elif 'id' in df.columns:  # Fallback to 'id' column if exists
            return str(df['id'].iloc[0])
        return Path(csv_path).stem  # Fallback to filename without extension
    except Exception as e:
        logger.error(f"Error reading sample ID from CSV: {str(e)}")
        return "unknown_sample"

def get_sample_dirs(base_dir: Path, sample_id: str) -> dict:
    """Create and return dictionary of sample-specific directories.
    
    Args:
        base_dir: Base directory path
        sample_id: Sample ID string
        
    Returns:
        Dictionary containing paths for each subdirectory
    """
    sample_dir = base_dir / sample_id
    dirs = {
        'sample': sample_dir,
        'models': sample_dir / 'models',
        'sgnn_output': sample_dir / 'sgnn_output',
        'experimental_data': sample_dir / 'experimental_data',
        'test_results': sample_dir / 'test_results'
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

# Add at the top with other imports
class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def save_json_with_tensor_handling(data, output_file):
    """Save data to JSON file with proper tensor handling."""
    with open(output_file, 'w') as f:
        json.dump(data, f, cls=TensorEncoder, indent=2)


def parse_arguments(config, args):
    """Parse command line arguments into config object.
    
    Args:
        config: Existing config object loaded from config files
        args: Command line arguments parsed by argparse
        
    Returns:
        Updated config object with command line arguments
    """
    # Convert all path-like arguments to strings
    if args.input_csv:
        config.input_csv = str(args.input_csv)
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.model_save_dir:
        config.model_save_dir = str(args.model_save_dir)
    if args.config_dir:
        config.config_dir = str(args.config_dir)
    if args.mol2mol_model_path:
        config.mol2mol_model_path = str(args.mol2mol_model_path)
    if args.mol2mol_vocab_path:
        config.mol2mol_vocab_path = str(args.mol2mol_vocab_path)
    if args.sgnn_gen_folder:
        config.sgnn_gen_folder = str(args.sgnn_gen_folder)
    if args.exp_data_path:
        config.exp_data_path = str(args.exp_data_path)
    if args.run_mode:
        config.run_mode = args.run_mode
    
    # Mol2Mol parameters
    if args.MF_delta_weight:
        config.MF_delta_weight = args.MF_delta_weight
    if args.tanimoto_filter:
        config.tanimoto_filter = args.tanimoto_filter
    if args.MF_max_trails:
        config.MF_max_trails = args.MF_max_trails
    if args.max_scaffold_generations:
        config.max_scaffold_generations = args.max_scaffold_generations
    
    # SGNN parameters
    if args.sgnn_gen_folder:
        config.sgnn_gen_folder = args.sgnn_gen_folder
    
    # MMST parameters
    if args.MF_generations:
        config.MF_generations = args.MF_generations
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.prediction_batch_size:
        config.prediction_batch_size = args.prediction_batch_size
    if args.improvement_cycles:
        config.improvement_cycles = args.improvement_cycles
    if args.IC_threshold:
        config.IC_threshold = args.IC_threshold
    if args.learning_rate:
        config.lr_pretraining = [args.learning_rate]
        config.lr_finetuning = [args.learning_rate]
    if args.mw_tolerance:
        config.mw_tolerance = args.mw_tolerance
    
    # Experimental workflow parameters
    if args.nmr_types:
        config.nmr_types = args.nmr_types
    if args.multinom_runs:
        config.multinom_runs = args.multinom_runs
    return config

def main():
    parser = argparse.ArgumentParser(description='MMST Structure Prediction Workflow')
    # Common parameters
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file containing a single SMILES and ID')
    parser.add_argument('--output_dir', required=True, help='Output directory for results and experiment data')
    parser.add_argument('--model_save_dir', required=True, help='Directory for saving fine-tuned models')
    parser.add_argument('--config_dir', required=True, help='Directory containing configuration files')
    parser.add_argument('--run_mode', choices=['test', 'improve', 'both'], default='both',
                        help='Run mode: test pre-trained model, run improvement cycle, or both')
    
    # Mol2Mol parameters
    parser.add_argument('--mol2mol_model_path', required=True, help='Path to Mol2Mol model')
    parser.add_argument('--mol2mol_vocab_path', required=True, help='Path to Mol2Mol vocabulary')
    parser.add_argument('--MF_delta_weight', type=int, default=100, help='Delta weight for Mol2Mol')
    parser.add_argument('--tanimoto_filter', type=float, default=0.2, help='Tanimoto filter threshold')
    parser.add_argument('--MF_max_trails', type=int, default=300, help='Maximum trails')
    parser.add_argument('--max_scaffold_generations', type=int, default=100, help='Maximum scaffold generations')
    
    # SGNN parameters
    parser.add_argument('--sgnn_gen_folder', required=True, help='SGNN generation folder')
    
    # MMST parameters
    parser.add_argument('--MF_generations', type=int, default=50, help='Number of analogues to generate')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of fine-tuning epochs')
    parser.add_argument('--prediction_batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--improvement_cycles', type=int, default=3, help='Number of improvement cycles to run')
    parser.add_argument('--IC_threshold', type=float, default=0.6, help='Performance threshold for improvement cycle')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for finetuning during improvement cycles')
    parser.add_argument('--mw_tolerance', type=float, default=0.5, help='Molecular weight tolerance in Da for filtering generated molecules')
    

    # Experimental workflow parameters
    parser.add_argument('--nmr_types', nargs='+', type=str, choices=['1H', '13C', 'HSQC', 'COSY'], default=['1H'],
                        help='Types of NMR data to use')
    parser.add_argument('--exp_data_path', 
                        help='Path to experimental data JSON file')
    parser.add_argument('--multinom_runs', type=int, default=10,
                        help='Number of multinomial sampling runs to generate molecules')
    
    args = parser.parse_args()
    
    try:
        # Get sample ID and create sample-specific directories
        sample_id = get_sample_id_from_csv(args.input_csv)
        sample_dirs = get_sample_dirs(Path(args.output_dir), sample_id)
        
        # Update paths to use sample-specific directories
        args.model_save_dir = str(sample_dirs['models'])
        args.sgnn_gen_folder = str(sample_dirs['sgnn_output'])
        
        # Load configurations
        itos, stoi, stoi_MF, itos_MF = load_json_dics()
        IR_config, config = load_configs(args.config_dir)
        
        # Update config with command line arguments
        config = parse_arguments(config, args)
        
        # Store original model paths
        original_checkpoint_path = config.checkpoint_path
        original_mt_model_path = config.MT_model_path

        all_run_summaries = []  # List to collect all run summaries
        num_runs = config.improvement_cycles

        # Run the entire process N times
        for run_num in range(0, num_runs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting complete run {run_num}/{num_runs}")
            logger.info(f"{'='*50}\n")
            
            # Reset model paths to original pre-trained model for each run
            config.checkpoint_path = original_checkpoint_path
            config.MT_model_path = original_mt_model_path

            # Step 1: Test on simulated data
            logger.info(f"Run {run_num} - Step 1: Testing on simulated data...")
            # performance, results_dict, results_dict_greedy = test_pretrained_model(
            #     config, stoi, itos, stoi_MF) 
            performance = 0.1
            results_dict = {}
            results_dict_greedy = {}
            logger.info(f"Run {run_num} - Simulated data performance: {performance}")


            # Create run-specific directory for this iteration's results
            run_dir = sample_dirs['test_results'] / f'run_{run_num}'
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"run_dir {run_dir}")

            # Save test results with run-specific path
            test_results = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'run_number': run_num,
                'performance': performance,
                'results': results_dict,
                'greedy_results': results_dict_greedy,
                'config': vars(config)
            }

            test_results_file = run_dir / 'simulated_test_results.json'
            save_json_with_tensor_handling(test_results, test_results_file)
            logger.info(f"test_results_file {test_results_file}")
            
                
            # Step 2: Check if performance meets threshold
            if performance >= config.IC_threshold:
                logger.info(f"Performance {performance} meets threshold {config.IC_threshold}")
                if config.exp_data_path:
                    # Define all NMR types
                    nmr_types = config.nmr_types
                    
                    # Prepare experimental data for each NMR type
                    exp_csv_paths = {}
                    for nmr_type in nmr_types:
                        exp_csv = sample_dirs['experimental_data'] / f"nmr_exp_{nmr_type}.csv"
                        prepare_experimental_data(
                            nmr_type, 
                            config.exp_data_path, 
                            exp_csv,
                            config.input_csv
                        )
                        exp_csv_paths[nmr_type] = str(exp_csv)
                    
                    # Save original input_csv
                    original_input = config.input_csv
                    
                    # Update config paths for each NMR type
                    config.csv_1H_path_SGNN = exp_csv_paths['1H']
                    config.csv_13C_path_SGNN = exp_csv_paths['13C']
                    config.csv_HSQC_path_SGNN = exp_csv_paths['HSQC']
                    config.csv_COSY_path_SGNN = exp_csv_paths['COSY']
                    config.csv_path_val = exp_csv_paths['1H']  # Set validation path to HSQC
                    config.pickle_file_path = ""  # Reset pickle file path
                    
                    # # Use HSQC data as main input for testing
                    # args.input_csv = exp_csv_paths['HSQC']
                    # config.input_csv = exp_csv_paths['HSQC']
                    
                    # Run on experimental data
                    logger.info("Running model on experimental data...")
                    exp_performance, exp_results, exp_results_greedy = run_experimental_data(
                        config, stoi, itos, stoi_MF)
                    logger.info(f"Experimental data performance: {exp_performance}")
                    
                    # Save experimental results in run-specific directory
                    exp_test_results = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'run_number': run_num,
                        'performance': exp_performance,
                        'results': exp_results,
                        'greedy_results': exp_results_greedy,
                        'model_save_path': str(sample_dirs['models']),
                        'improvement_cycle': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'config': vars(config)
                    }

                    exp_results_file = run_dir / 'experimental_results_wo_IC.json'
                    save_json_with_tensor_handling(exp_test_results, exp_results_file)

                
                # # Restore original input
                # args.input_csv = original_input
                # config.input_csv = original_input
            
            else:
                logger.info(f"Performance {performance} below threshold {config.IC_threshold}")
                logger.info("Starting improvement cycle...")
                
                # Create directories for improvement cycle within the run directory
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create improvement cycle directory within run directory
                ic_dir = run_dir / f"improvement_cycle_{current_time}"
                ic_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"ic_dir {ic_dir}")

                # Create model directory with run number and timestamp
                model_save_path = sample_dirs['models'] / f"run_{run_num}_model_{current_time}"
                model_save_path.mkdir(parents=True, exist_ok=True)
                config.model_save_dir = str(model_save_path)

                # Update MOL2MOL config parameters to use the input file
                config.MF_csv_source_folder_location = str(sample_dirs['sample'])
                config.MF_csv_source_file_name = 'mmst_input'

                # Update learning rate for improvement cycle
                # config.lr_pretraining = 3e-4

                # Run improvement cycle
                logger.info(f"Start IC")
                final_performance, ic_results = run_improvement_cycle(config, stoi, itos, stoi_MF, itos_MF, IR_config)
                logger.info(f"Final improvement cycle performance: {final_performance}")

                # Save improvement cycle results within run directory
                ic_result_paths = []
                for i, result in enumerate(ic_results, 1):
                    ic_results_file = ic_dir / f'cycle_{i}_results.json'
                    save_json_with_tensor_handling(result, ic_results_file)
                    ic_result_paths.append(str(ic_results_file))

                # Save summary of improvement cycle
                ic_summary = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'run_number': run_num,
                    'initial_performance': performance,
                    'final_performance': final_performance,
                    'model_save_path': str(model_save_path),
                    'cycle_results': ic_result_paths
                }
                save_json_with_tensor_handling(ic_summary, ic_dir / 'improvement_cycle_summary.json')

               # if final_performance >= config.IC_threshold and config.exp_data_path:   ###########
                # If improved performance meets threshold, run on experimental data
                # Define all NMR types
                nmr_types = config.nmr_types

                # Prepare experimental data for each NMR type
                exp_csv_paths = {}
                for nmr_type in nmr_types:
                    exp_csv = sample_dirs['experimental_data'] / f"nmr_exp_{nmr_type}.csv"
                    prepare_experimental_data(
                        nmr_type, 
                        config.exp_data_path, 
                        exp_csv,
                        config.input_csv
                    )
                    exp_csv_paths[nmr_type] = str(exp_csv)

                # # Save original input
                # original_input = args.input_csv

                # Update config paths for each NMR type
                config.csv_1H_path_SGNN = exp_csv_paths['1H']
                config.csv_13C_path_SGNN = exp_csv_paths['13C']
                config.csv_HSQC_path_SGNN = exp_csv_paths['HSQC']
                config.csv_COSY_path_SGNN = exp_csv_paths['COSY']
                config.csv_path_val = exp_csv_paths['1H']  # Set validation path to HSQC
                config.pickle_file_path = ""  # Reset pickle file path

                # # Use HSQC data as main input for testing
                # args.input_csv = exp_csv_paths['HSQC']
                # config.input_csv = exp_csv_paths['HSQC']

                # Run on experimental data with improved model
                logger.info("Running improved model on experimental data...")
                exp_performance, exp_results, exp_results_greedy = run_experimental_data(
                    config, stoi, itos, stoi_MF)
                logger.info(f"Experimental data performance with improved model: {exp_performance}")

                # Save experimental results
                exp_test_results = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'run_number': run_num,
                    'performance': exp_performance,
                    'results': exp_results,
                    'greedy_results': exp_results_greedy,
                    'model_save_path': str(model_save_path),
                    'improvement_cycle_dir': str(ic_dir),
                    'config': vars(config)
                }

                exp_results_file = ic_dir / 'experimental_results_after_IC.json'
                save_json_with_tensor_handling(exp_test_results, exp_results_file)
                logger.info(f"Saved experimental results to {exp_results_file}")

                # Save individual run results
                run_results = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'run_number': run_num,
                    'final_performance': final_performance,
                    'exp_results_file': str(exp_results_file),
                    'test_results_file': str(test_results_file),
                    'model_save_path': str(model_save_path),
                    'improvement_cycle_dir': str(ic_dir),
                    'improvement_cycle_results': ic_result_paths,
                }

                run_results_file = run_dir / f'run_{run_num}_results.json'
                with open(run_results_file, 'w') as f:
                    json.dump(run_results, f, indent=4)
                logger.info(f"Saved run {run_num} results to {run_results_file}")

                all_run_summaries.append(run_results)

        # After all runs are complete, create final results file
        final_results_file = sample_dirs['test_results'] / 'mmst_final_results.json'
        final_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_runs': num_runs,
            'runs': all_run_summaries,
            'test_results_dir': str(sample_dirs['test_results'])
        }
        
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"Created final results file at {final_results_file}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    sys.exit(main())
