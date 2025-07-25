
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List, Optional
import shutil


# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import SGNN utilities
import utils_MMT.data_generation_v15_4 as dg


class Config:
    """Configuration class that allows dot notation access to parameters."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        """Called when an attribute lookup has not found the attribute in the usual places."""
        raise AttributeError(f"Configuration has no attribute '{name}'. Available attributes: {', '.join(self.__dict__.keys())}")
    
    def to_dict(self):
        """Convert config back to dictionary if needed."""
        return self.__dict__


def simulate_nmr_data(config: Config) -> Config:
    """
    Simulate NMR data for molecules.
    
    Args:
        config: Configuration object containing simulation parameters
        
    Returns:
        Updated config
    """
    # Create simulation output directory if it doesn't exist
    output_dir = Path(config.SGNN_gen_folder_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique run ID if not provided
    if not hasattr(config, 'ran_num'):
        config.ran_num = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run directory
    run_dir = output_dir / f"syn_{config.ran_num}"
    run_dir.mkdir(exist_ok=True)
    
    # Set paths in config
    config.SGNN_gen_folder_path = str(run_dir)
    
    # Run NMR data generation
    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)
    
    # After data generation, copy files to temp folder with expected naming
    temp_folder = Path(config.csv_SMI_targets).parent  # This will be the _temp_folder
    
    # Copy files with expected naming convention
    file_mapping = {
        '1H': csv_1H_path,
        '13C': csv_13C_path,
        'COSY': csv_COSY_path,
        'HSQC': csv_HSQC_path
    }
    
    for nmr_type, source_path in file_mapping.items():
        target_path = temp_folder / f"nmr_prediction_{nmr_type}.csv"
        shutil.copy2(source_path, target_path)
        print(f"Copied {nmr_type} NMR predictions to {target_path}")

    return config

def read_input_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads and validates input CSV file.
    
    Args:
        csv_path (str): Path to input CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing SMILES and sample-id columns
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)
    required_cols = ['SMILES', 'sample-id']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")
        
    return df

import random
def example_usage():
    """Example of how to use the NMR simulation function."""
    # Example configuration
    random_number = int(datetime.now().timestamp())
    config_dict  = {
        'SGNN_gen_folder_path': '/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/experiments',
        "SGNN_csv_save_folder":'/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/experiments',
        'ran_num': str(random_number),
        "SGNN_size_filter":550,
        'csv_SMI_targets': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/_temp_folder/current_molecule.csv",
        'SGNN_csv_gen_smi': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/_temp_folder/current_molecule.csv"  # Add this line
    }
    
    # Convert dictionary to Config object
    config = Config(**config_dict)

    # Read input data
    input_df = read_input_csv(config.csv_SMI_targets)  # Now we can use dot notation
    
    # Run simulation
    config = simulate_nmr_data(config)
    
    # Example of accessing results
    #print(f"Generated {len(combined_df)} simulated NMR spectra")
    
    # Paths to generated files are stored in config:
    print(f"Finished simulating NMR data")

if __name__ == "__main__":
    example_usage()