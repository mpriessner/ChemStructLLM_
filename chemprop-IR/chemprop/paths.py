"""Configuration for model and data paths.

This module provides centralized path configuration for the chemprop-IR project.
It supports overriding paths via environment variables and provides helper functions
for path resolution.
"""
import os
import logging

# Set up a logger for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the models directory in the parent directory
# Can be overridden with MODELS_BASE_DIR environment variable
MODELS_BASE_DIR = os.environ.get('MODELS_BASE_DIR', 
                                os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models'))

# Path to the chemprop-IR models
# Can be overridden with CHEMPROP_IR_MODEL_DIR environment variable
CHEMPROP_IR_MODEL_DIR = os.environ.get('CHEMPROP_IR_MODEL_DIR', 
                                     os.path.join(MODELS_BASE_DIR, 'chemprop-IR'))

# Log the paths being used
logger.info(f"Using BASE_DIR: {BASE_DIR}")
logger.info(f"Using MODELS_BASE_DIR: {MODELS_BASE_DIR}")
logger.info(f"Using CHEMPROP_IR_MODEL_DIR: {CHEMPROP_IR_MODEL_DIR}")

# Create directories if they don't exist
try:
    os.makedirs(CHEMPROP_IR_MODEL_DIR, exist_ok=True)
    logger.info("Successfully created model directories")
except Exception as e:
    logger.error(f"Error creating directories: {str(e)}")

def resolve_model_path(model_file, model_idx=None):
    """Resolve a model file path by checking multiple possible locations.
    
    Args:
        model_file (str): Base name of the model file or relative path
        model_idx (int, optional): Model index to append to directory name
        
    Returns:
        str: Full path to the model file if found, None otherwise
    """
    if model_idx is not None:
        model_dir = f'model_{model_idx}'
    else:
        model_dir = ''
        
    # List of possible locations to check
    possible_paths = [
        os.path.join(CHEMPROP_IR_MODEL_DIR, model_dir, model_file),
        model_file  # In case it's already an absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            return path
            
    logger.warning(f"Model file '{model_file}' not found in any of the expected locations")
    return None

def get_save_dir(original_save_dir, model_idx=None):
    """Get the save directory for a model.
    
    Args:
        original_save_dir (str): Original save directory specified in args
        model_idx (int, optional): Model index to append to directory name
        
    Returns:
        str: Path to the save directory
    """
    # If original_save_dir is an absolute path, use it
    if os.path.isabs(original_save_dir):
        save_dir = original_save_dir
    else:
        # Otherwise, use the new model directory
        save_dir = os.path.join(CHEMPROP_IR_MODEL_DIR, original_save_dir)
    
    # Append model_idx if provided
    if model_idx is not None:
        save_dir = os.path.join(save_dir, f'model_{model_idx}')
        
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    return save_dir
