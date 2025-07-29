#!/usr/bin/env python
import sys
import os

# Remove any existing molbart from sys.modules to avoid conflicts
if 'molbart' in sys.modules:
    del sys.modules['molbart']

# Get the directory where this script is located and build relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from scripts -> agents -> LLM_Structure_Elucidator -> ChemStructLLM_ (containing chemformer_public)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, parent_dir)

# Import chemformer_public.molbart and make it available as 'molbart' for internal imports
import chemformer_public.molbart as molbart
sys.modules['molbart'] = molbart

# Now import from your local version
from chemformer_public.molbart.models import Chemformer
import hydra
import omegaconf
import pandas as pd
import sys
import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run retrosynthesis predictions")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file with target SMILES")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for predictions")
    parser.add_argument("--n_beams", type=int, default=50, help="Number of beams for beam search")
    parser.add_argument("--n_unique_beams", type=int, default=-1, help="Number of unique beams to return. Use -1 for no limit")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (higher = more diverse)")
    parser.add_argument("--sampling_method", type=str, default="beam", choices=["beam", "multinomial"], help="Sampling method to use")
    return parser.parse_args()

def create_config(args):
    """Create Chemformer configuration."""
    config = {
        'data_path': args.input_file,
        'vocabulary_path': args.vocab_path,
        'model_path': args.model_path,
        'task': 'backward_prediction',
        'output_sampled_smiles': args.output_file,
        'batch_size': args.batch_size,
        'n_beams': args.n_beams,
        "n_unique_beams": None if args.n_unique_beams == -1 else args.n_unique_beams,
        'n_gpus': 1 if torch.cuda.is_available() else 0,
        'train_mode': 'eval',
        'model_type': 'bart',
        'datamodule': ['SynthesisDataModule'],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "temperature": args.temperature,
        "sampling_method": args.sampling_method,
        "sampling_alg": args.sampling_method
    }
    return OmegaConf.create(config)

def main():
    """Main function to run retrosynthesis predictions."""
    try:
        # Parse arguments
        args = parse_arguments()
        # logger.info("Creating Chemformer configuration")
        
        # Create config
        config = create_config(args)
        logger.info(f"Configuration created: {config}")
        
        # Initialize model
        logger.info("Initializing Chemformer model")
        chemformer = Chemformer(config)
        
        # Run prediction
        logger.info("Running predictions")
        smiles, log_lhs, target_smiles = chemformer.predict(dataset='full')
        
        logger.info(f"smiles {smiles}")
        logger.info(f"target_smiles {target_smiles}")

        # Save predictions to CSV
        logger.info("Saving predictions to CSV")
        predictions_df = pd.DataFrame({
            'target_smiles': target_smiles,
            'predicted_smiles': [s[0].item() if isinstance(s, (list, np.ndarray)) and len(s) > 0 else '' for s in smiles],
            'log_likelihood': [float(l[0]) if isinstance(l, (list, np.ndarray)) and len(l) > 0 else 0.0 for l in log_lhs],
            'all_predictions': [';'.join(map(str, s)) if isinstance(s, (list, np.ndarray)) else '' for s in smiles],
            'all_log_likelihoods': [';'.join(map(str, l)) if isinstance(l, (list, np.ndarray)) else '' for l in log_lhs]
        })
        print(predictions_df)
        print(f"target_smiles: {target_smiles}")
        # # Debug logging
        # logger.info(f"DataFrame shape: {predictions_df.shape}")
        # logger.info(f"DataFrame columns: {predictions_df.columns}")
        # if len(predictions_df) > 0:
        #     logger.info("First row of predictions:")
        #     logger.info(predictions_df.iloc[0].to_dict())
        
        predictions_df.to_csv(args.output_file, index=False)
        # logger.info(f"Predictions completed. Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in retrosynthesis pipeline: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()