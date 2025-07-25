"""Configuration for model and data paths.

This module provides centralized path configuration for the deep-molecular-optimization project.
It supports overriding paths via environment variables and provides helper functions
for path resolution.
"""
import os
import logging

# Set up a logger for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the models directory in the parent directory
# Can be overridden with MODELS_BASE_DIR environment variable
MODELS_BASE_DIR = os.environ.get('MODELS_BASE_DIR', 
                                os.path.join(os.path.dirname(BASE_DIR), 'models'))

# Path to the mol2mol models
# Can be overridden with MOL2MOL_MODEL_DIR environment variable
MOL2MOL_MODEL_DIR = os.environ.get('MOL2MOL_MODEL_DIR', 
                                  os.path.join(MODELS_BASE_DIR, 'mol2mol', 'Alessandro_big'))

# Path to save checkpoints during training
# Can be overridden with CHECKPOINT_DIR environment variable
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 
                               os.path.join(MODELS_BASE_DIR, 'checkpoints'))

# Log the paths being used
logger.info(f"Using BASE_DIR: {BASE_DIR}")
logger.info(f"Using MODELS_BASE_DIR: {MODELS_BASE_DIR}")
logger.info(f"Using MOL2MOL_MODEL_DIR: {MOL2MOL_MODEL_DIR}")
logger.info(f"Using CHECKPOINT_DIR: {CHECKPOINT_DIR}")

# Create directories if they don't exist
try:
    os.makedirs(MOL2MOL_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger.info("Successfully created model directories")
except Exception as e:
    logger.error(f"Error creating directories: {str(e)}")

def resolve_model_path(model_file, epoch=None):
    """Resolve a model file path by checking multiple possible locations.
    
    Args:
        model_file (str): Base name of the model file or relative path
        epoch (int, optional): Epoch number to append to filename
        
    Returns:
        str: Full path to the model file if found, None otherwise
    """
    if epoch is not None:
        model_file = f"model_{epoch}.pt" if not model_file else model_file
        
    # List of possible locations to check
    possible_paths = [
        os.path.join(MOL2MOL_MODEL_DIR, 'checkpoint', model_file),
        os.path.join(CHECKPOINT_DIR, model_file),
        os.path.join(BASE_DIR, 'experiments', model_file),
        model_file  # In case it's already an absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            return path
            
    logger.warning(f"Model file '{model_file}' not found in any of the expected locations")
    return None

def resolve_data_path(data_file):
    """Resolve a data file path by checking multiple possible locations.
    
    Args:
        data_file (str): Base name of the data file or relative path
        
    Returns:
        str: Full path to the data file if found, None otherwise
    """
    # List of possible locations to check
    possible_paths = [
        os.path.join(MOL2MOL_MODEL_DIR, data_file),
        os.path.join(BASE_DIR, 'data', data_file),
        data_file  # In case it's already an absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found data at: {path}")
            return path
            
    logger.warning(f"Data file '{data_file}' not found in any of the expected locations")
    return None
