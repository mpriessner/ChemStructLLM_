import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List, Optional
import shutil
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import data generation module
try:
    import utils_MMT.data_generation_v15_4 as dg
    logger.info("Successfully imported data generation module")
except ImportError as e:
    logger.error(f"Failed to import data generation module: {e}")
    raise


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
    logger.info("Starting NMR data simulation")
    
    # Create simulation output directory if it doesn't exist
    output_dir = Path(config.SGNN_gen_folder_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Generate unique run ID if not provided
    if not hasattr(config, 'ran_num'):
        config.ran_num = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {config.ran_num}")
    
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
        logger.info(f"Copied {nmr_type} NMR predictions to {target_path}")
        # Clean up source file after copying
        try:
            Path(source_path).unlink()
            logger.info(f"Cleaned up source file: {source_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up source file {source_path}: {str(e)}")
    
    # Clean up the temporary syn_ directory after copying results
    try:
        shutil.rmtree(run_dir)
        logger.info(f"Cleaned up temporary directory: {run_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {run_dir}: {str(e)}")
        

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

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NMR simulation')
    parser.add_argument('--input_file', required=True, help='Path to input CSV file')
    args = parser.parse_args()
    
    # Get the directory containing the input file
    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    
    # Configuration using input file location
    random_number = int(datetime.now().timestamp())
    config_dict = {
        'SGNN_gen_folder_path': input_dir,
        "SGNN_csv_save_folder": input_dir,
        'ran_num': str(random_number),
        "SGNN_size_filter": 550,
        'csv_SMI_targets': args.input_file,
        'SGNN_csv_gen_smi': args.input_file
    }
    
    # Convert dictionary to Config object
    config = Config(**config_dict)
    
    try:
        # Read input data
        logger.info(f"Reading input file: {args.input_file}")
        input_df = read_input_csv(config.csv_SMI_targets)
        logger.info(f"Input data: {input_df}")
        
        # Run simulation
        logger.info("Starting NMR simulation...")
        config = simulate_nmr_data(config)
        logger.info("NMR simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)
        sys.exit(1)