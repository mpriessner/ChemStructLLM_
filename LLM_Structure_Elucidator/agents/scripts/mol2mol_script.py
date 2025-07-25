import sys
import os
import json
import random
from pathlib import Path
import pandas as pd
import argparse
from types import SimpleNamespace
from typing import Dict, Any, Optional
from rdkit import Chem

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import Mol2Mol utilities
import utils_MMT.execution_function_v15_4 as ef

def load_json_dics(config_dir):
    """Load JSON dictionaries for tokenization"""
    with open(os.path.join(config_dir, 'itos.json'), 'r') as f:
        itos = json.load(f)
    with open(os.path.join(config_dir, 'stoi.json'), 'r') as f:
        stoi = json.load(f)
    with open(os.path.join(config_dir, 'stoi_MF.json'), 'r') as f:
        stoi_MF = json.load(f)
    with open(os.path.join(config_dir, 'itos_MF.json'), 'r') as f:
        itos_MF = json.load(f)    
    return itos, stoi, stoi_MF, itos_MF

def setup_molformer_config(params):
    """Create config namespace for Molformer"""
    config = {
        "MF_max_trails": params.max_trials,
        "MF_tanimoto_filter": params.tanimoto_filter,
        "MF_filter_higher": 1,  # True = generate more similar molecules
        "MF_delta_weight": params.delta_weight,
        "MF_generations": params.num_generations,
        "MF_model_path": params.model_path,
        "MF_vocab": params.vocab_path,
        "MF_csv_source_folder_location": os.path.dirname(params.input_csv),
        "MF_csv_source_file_name": Path(params.input_csv).stem,
        "MF_methods": ["MMP"], #scaffold , MMP
        "max_scaffold_generations": params.max_scaffold_generations,
    }
    
    return SimpleNamespace(**config)

def run_molformer(params):
    """Main function to run Molformer"""
    # Create output directory
    os.makedirs(params.output_dir, exist_ok=True)
    
    # Setup lock files
    running_lock = os.path.join(params.output_dir, "mol2mol_running.lock")
    complete_lock = os.path.join(params.output_dir, "mol2mol_complete.lock")
    
    try:
        # Load dictionaries
        itos, stoi, stoi_MF, itos_MF = load_json_dics(params.config_dir)
        
        # Setup configuration
        config = setup_molformer_config(params)
        
        # Run Molformer generation
        config, results_dict = ef.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)
        
        # Convert results to DataFrame and save
        df_results = pd.DataFrame.from_dict(results_dict, orient='index').transpose()
        output_file = os.path.join(params.output_dir, "generated_molecules.csv")
        df_results.to_csv(output_file, index=False)
        
        print(f"Successfully generated molecules. Results saved to: {output_file}")
        
        # Signal completion by creating complete lock and removing running lock
        if os.path.exists(running_lock):
            os.remove(running_lock)
        with open(complete_lock, 'w') as f:
            f.write('done')
        
        return df_results, output_file
        
    except Exception as e:
        print(f"Error occurred during molecule generation: {str(e)}")
        # Clean up lock files in case of error
        if os.path.exists(running_lock):
            os.remove(running_lock)
        if os.path.exists(complete_lock):
            os.remove(complete_lock)
        raise

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string can be parsed by RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main(args):
    """
    Main function for Mol2Mol generation
    """
    # Input file validation
    input_file = Path(args.input_csv)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    # Read and validate standardized input
    df = pd.read_csv(input_file)
    if 'SMILES' not in df.columns or 'sample-id' not in df.columns:
        raise ValueError("Input file must contain SMILES and sample-id columns")
    
    # Validate SMILES strings
    invalid_smiles = []
    for idx, row in df.iterrows():
        if not validate_smiles(row['SMILES']):
            invalid_smiles.append((idx, row['SMILES'], row['sample-id']))
    
    if invalid_smiles:
        error_msg = "Invalid SMILES strings found:\n"
        for idx, smiles, sample_id in invalid_smiles:
            error_msg += f"Row {idx}: SMILES='{smiles}', sample-id='{sample_id}'\n"
        raise ValueError(error_msg)
    
    # Run Molformer generation
    df_results, output_file = run_molformer(args)
    print(f"Results saved to: {output_file}")
    print(f"Generated {len(df_results)} molecule analogues")
    print(f"Results saved to: {output_file}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mol2Mol generation')
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--config_dir', required=True, help='Directory containing configuration files')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--vocab_path', required=True, help='Path to vocabulary file')
    parser.add_argument('--delta_weight', type=int, default=30, help='Delta weight parameter')
    parser.add_argument('--tanimoto_filter', type=float, default=0.2, help='Tanimoto filter threshold')
    parser.add_argument('--num_generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--max_trials', type=int, default=100, help='Maximum number of trials')
    parser.add_argument('--max_scaffold_generations', type=int, default=10, help='Maximum scaffold generations')
    
    args = parser.parse_args()
    main(args)
