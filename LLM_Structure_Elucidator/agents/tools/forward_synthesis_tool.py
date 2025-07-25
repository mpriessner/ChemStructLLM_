"""Tool for generating forward synthesis predictions using Chemformer."""
import os
import logging
import shutil
import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
FORWARD_DIR = BASE_DIR / "_temp_folder" / "forward_output"
TEMP_DIR = BASE_DIR / "_temp_folder"
SBATCH_SCRIPT = SCRIPTS_DIR / "chemformer_forward_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "chemformer_forward_local.sh"

# Constants for Chemformer execution
FORWARD_OUTPUT_CHECK_INTERVAL = 5  # seconds
FORWARD_OUTPUT_TIMEOUT = 600  # 10 minutes
FORWARD_OUTPUT_PATTERN = "forward_predictions_{}.csv"  # Will be formatted with timestamp
FORWARD_INPUT_FILENAME = "forward_reactants.txt"  # Input filename

class ForwardSynthesisTool:
    """Tool for generating forward synthesis predictions using Chemformer."""
    
    def __init__(self):
        """Initialize the Forward Synthesis tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        self.forward_dir = FORWARD_DIR
        
        # Create directories if they don't exist
        self.forward_dir.mkdir(exist_ok=True)
        
        # Add intermediate results directory
        self.temp_dir = TEMP_DIR
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Validate script existence
        if not SBATCH_SCRIPT.exists():
            raise FileNotFoundError(f"Required SBATCH script not found at {SBATCH_SCRIPT}")
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required local script not found at {LOCAL_SCRIPT}")

        # Validate environment
        try:
            import torch
            if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                logging.warning("CUDA not available for local execution. SLURM execution will be forced.")
        except ImportError:
            logging.warning("PyTorch not found. Please ensure the chemformer environment is activated.")

    # def _normalize_smiles(self, smiles: str) -> str:
    #     """Normalize SMILES string to canonical form."""
    #     try:
    #         from rdkit import Chem
    #         # Handle disconnected structures by splitting on '.'
    #         parts = smiles.split('.')
    #         normalized_parts = []
    #         for part in parts:
    #             mol = Chem.MolFromSmiles(part.strip())
    #             if mol is not None:
    #                 normalized_parts.append(Chem.MolToSmiles(mol, canonical=True))
    #         if normalized_parts:
    #             return '.'.join(normalized_parts)
    #     except ImportError:
    #         logging.warning("RDKit not available for SMILES normalization")
    #     except Exception as e:
    #         logging.warning(f"Error normalizing SMILES {smiles}: {str(e)}")
    #     return smiles.strip()

    # def _prepare_input_from_context(self, context: Dict[str, Any]) -> tuple[Path, Dict[str, list]]:
    #     """Prepare input file from context and return mapping of predictions to source molecules.
        
    #     Args:
    #         context: Dictionary containing molecule data and flags
            
    #     Returns:
    #         Tuple of:
    #         - Path to the created input file
    #         - Dictionary mapping molecule IDs to their starting material indices in the input file
    #     """
    #     try:
    #         # Initialize variables
    #         smiles_list = []
    #         input_file = self.temp_dir / FORWARD_INPUT_FILENAME
    #         molecule_mapping = {}  # Maps molecule IDs to their starting material indices
    #         current_index = 0
            
    #         # Get molecules from master JSON file
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if master_data_path.exists():
    #             with open(master_data_path, 'r') as f:
    #                 master_data = json.load(f)
    #             # Extract starting materials from master data
    #             for molecule_id, molecule_data in master_data.items():
    #                 if 'starting_smiles' in molecule_data:
    #                     start_idx = current_index
    #                     for starting_smiles in molecule_data['starting_smiles']:
    #                         if isinstance(starting_smiles, list):
    #                             for smiles in starting_smiles:
    #                                 smiles_list.append(smiles.strip())
    #                                 current_index += 1
    #                         elif isinstance(starting_smiles, str):
    #                             smiles_list.append(starting_smiles.strip())
    #                             current_index += 1
    #                     # Store the range of indices for this molecule's starting materials
    #                     if current_index > start_idx:
    #                         molecule_mapping[molecule_id] = (start_idx, current_index)
    #             # logging.info(f"Extracted {len(smiles_list)} starting materials from master JSON")
    #         else:
    #             logging.warning("Master JSON file not found")
            
    #         # If no starting materials found in master JSON, try context as fallback
    #         if not smiles_list and context.get('current_molecule'):
    #             current_molecule = context['current_molecule']
    #             if isinstance(current_molecule, dict):
    #                 # Try to get starting materials from current molecule
    #                 if 'starting_smiles' in current_molecule:
    #                     materials = current_molecule['starting_smiles']
    #                     if isinstance(materials, list):
    #                         for material in materials:
    #                             smiles_list.append(material.strip())
    #                     elif isinstance(materials, str):
    #                         smiles_list.append(materials.strip())
    #             elif isinstance(current_molecule, str):
    #                 # If it's just a SMILES string, use it as is
    #                 smiles_list.append(current_molecule.strip())
            
    #         # Validate SMILES list
    #         if not smiles_list:
    #             raise ValueError("No valid starting materials found in master JSON or context")
            
    #         # Write SMILES to input file
    #         with open(input_file, 'w') as f:
    #             for smiles in smiles_list:
    #                 f.write(f"{smiles}\n")
            
    #         # logging.info(f"Prepared input file with {len(smiles_list)} starting materials")
    #         return input_file, molecule_mapping
            
    #     except Exception as e:
    #         logging.error(f"Error preparing input file: {str(e)}")
    #         raise

    # async def _update_master_data(self, predictions_df: pd.DataFrame, molecule_mapping: Dict[str, tuple]) -> None:
    #     """Update the master data file with forward synthesis predictions.
        
    #     Args:
    #         predictions_df: DataFrame containing forward synthesis predictions
    #         molecule_mapping: Dictionary mapping molecule IDs to their prediction indices
    #     """
    #     try:
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if not master_data_path.exists():
    #             logging.error("Master data file not found")
    #             return
             
    #         # Read current master data
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         # Convert predictions to list for easier processing
    #         predictions_list = predictions_df.to_dict('records')
            
    #         updated = False
    #         # Update each molecule's data with forward synthesis predictions
    #         for molecule_id, (start_idx, end_idx) in molecule_mapping.items():
    #             if molecule_id in master_data:
    #                 # Get all predictions for this molecule's starting materials
    #                 molecule_predictions = predictions_list[start_idx:end_idx]
                    
    #                 # Initialize forward_predictions if not exists
    #                 if 'forward_predictions' not in master_data[molecule_id]:
    #                     master_data[molecule_id]['forward_predictions'] = []
                    
    #                 # Process predictions for each starting material
    #                 for pred in molecule_predictions:
    #                     # Create prediction entry exactly matching CSV format
    #                     prediction = {
    #                         'starting_material': pred['target_smiles'],  
    #                         'predicted_smiles': pred['predicted_smiles'],
    #                         'log_likelihood': float(pred['log_likelihood']),
    #                         'all_predictions': pred['all_predictions'].split(';'),
    #                         'all_log_likelihoods': [float(l.strip()) for l in pred['all_log_likelihoods'].split(';')]
    #                     }
                        
    #                     # Check if this prediction already exists
    #                     exists = False
    #                     for existing_pred in master_data[molecule_id]['forward_predictions']:
    #                         if (existing_pred['starting_material'] == prediction['starting_material'] and
    #                             existing_pred['predicted_smiles'] == prediction['predicted_smiles']):
    #                             exists = True
    #                             break
                        
    #                     if not exists:
    #                         master_data[molecule_id]['forward_predictions'].append(prediction)
    #                         updated = True
    #                         # logging.info(f"Added forward prediction for sample {molecule_id} starting material {pred['target_smiles']}")
            
    #         if updated:
    #             # Write updated data back to file
    #             with open(master_data_path, 'w') as f:
    #                 json.dump(master_data, f, indent=2)
    #             logging.info("Successfully updated master data with forward synthesis predictions")
    #         else:
    #             logging.warning("No new predictions to add to master data")
                
    #     except Exception as e:
    #         logging.error(f"Error updating master data: {str(e)}")
    #         raise

    async def predict_forward_synthesis(self, molecule_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            # Get sample_id from context or molecule_data
            sample_id = None
            if context and 'sample_id' in context:
                sample_id = context['sample_id']
            elif isinstance(molecule_data, dict) and 'sample_id' in molecule_data:
                sample_id = molecule_data['sample_id']
            
            if not sample_id:
                raise ValueError("No sample_id provided in context or molecule_data")

            # Load or create intermediate file
            intermediate_data = self._load_or_create_intermediate(sample_id, molecule_data)
            
            # Check if forward synthesis predictions already exist
            if ('molecule_data' in intermediate_data and 
                'forward_predictions' in intermediate_data['molecule_data'] and
                intermediate_data['molecule_data']['forward_predictions']):
                logging.info(f"Forward synthesis predictions already exist for sample {sample_id}")
                return {
                    'status': 'success',
                    'message': 'Forward synthesis predictions already exist',
                    'predictions': intermediate_data['molecule_data']['forward_predictions']
                }

            # Get starting materials from molecule data
            if not isinstance(intermediate_data.get('molecule_data', {}), dict):
                intermediate_data['molecule_data'] = {}
            
            starting_smiles = intermediate_data['molecule_data'].get('starting_smiles')
            if not starting_smiles:
                starting_smiles = molecule_data.get('starting_smiles')  # Try getting from original molecule_data
                if starting_smiles:
                    intermediate_data['molecule_data']['starting_smiles'] = starting_smiles
                else:
                    raise ValueError("No starting materials found in molecule data")
            
            # Create input file with starting materials
            input_file = self.temp_dir / f"{sample_id}_input.txt"
            with open(input_file, 'w') as f:
                for smiles in starting_smiles:
                    f.write(f"{smiles}\n")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.forward_dir / FORWARD_OUTPUT_PATTERN.format(timestamp)
            
            # Check CUDA availability for local execution
            use_slurm = context.get('use_slurm', False)
            if not use_slurm:
                try:
                    import torch
                    if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                        logging.warning("CUDA not available. Switching to SLURM execution.")
                        use_slurm = True
                except ImportError:
                    logging.warning("PyTorch not found. Switching to SLURM execution.")
                    use_slurm = True
            
            logging.info(f"Running forward synthesis prediction using {'SLURM' if use_slurm else 'local'} execution")
            
            if use_slurm:
                try:
                    # Execute using SLURM
                    logging.info("Running forward synthesis with SLURM")
                    job_id = await execute_sbatch(
                        str(SBATCH_SCRIPT),
                        f"--input_file={input_file}",
                        f"--output_file={output_file}"
                    )
                    logging.info(f"SLURM job submitted with ID: {job_id}")
                    
                    success = await wait_for_job_completion(job_id)
                    if not success:
                        logging.error("SLURM job failed during execution")
                        return {
                            'status': 'error',
                            'message': 'SLURM job failed during execution'
                        }
                    
                    # Add a small delay to ensure file is fully written
                    logging.info("SLURM job completed, waiting for file system sync...")
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logging.error(f"Error during SLURM execution: {str(e)}")
                    return {
                        'status': 'error',
                        'message': f'Error during SLURM execution: {str(e)}'
                    }
                
                # Check if the output file exists and has content
                try:
                    if not output_file.exists():
                        error_msg = f"Output file not found at {output_file} after SLURM job completion"
                        logging.error(error_msg)
                        return {
                            'status': 'error',
                            'message': error_msg
                        }
                    
                    file_size = output_file.stat().st_size
                    logging.info(f"Output file exists with size: {file_size} bytes")
                    
                    if file_size == 0:
                        error_msg = "Output file is empty after SLURM job completion"
                        logging.error(error_msg)
                        return {
                            'status': 'error',
                            'message': error_msg
                        }
                except Exception as e:
                    logging.error(f"Error checking output file: {str(e)}")
                    return {
                        'status': 'error',
                        'message': f'Error checking output file: {str(e)}'
                    }
            else:
                # Execute locally
                try:
                    logging.info("Running forward synthesis locally")
                    process = await asyncio.create_subprocess_exec(
                        str(LOCAL_SCRIPT),
                        f"--input_file={input_file}",
                        f"--output_file={output_file}",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        error_msg = f"Local execution failed with return code {process.returncode}: {stderr.decode()}"
                        logging.error(error_msg)
                        return {'status': 'error', 'message': error_msg}
                        
                except Exception as e:
                    error_msg = f"Error during local execution: {str(e)}"
                    logging.error(error_msg)
                    return {'status': 'error', 'message': error_msg}
            
            # Wait for output file
            if not await self._wait_for_output(output_file):
                error_msg = f"Timeout waiting for output file at {output_file}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Read predictions from output file
            try:
                predictions_df = pd.read_csv(output_file)
                
                # Process predictions maintaining the same structure
                forward_predictions = []
                for _, row in predictions_df.iterrows():
                    # Get all predictions and their log likelihoods
                    all_preds = [p.strip() for p in row['all_predictions'].split(';') if p.strip()]
                    all_logs = [float(l.strip()) for l in row['all_log_likelihoods'].split(';') if l.strip()]
                    
                    prediction = {
                        'starting_material': row['target_smiles'],
                        'predicted_smiles': row['predicted_smiles'],
                        'log_likelihood': float(row['log_likelihood']),
                        'all_predictions': all_preds,
                        'all_log_likelihoods': all_logs
                    }
                    forward_predictions.append(prediction)
                
                # Store predictions directly in molecule_data
                intermediate_data['molecule_data']['forward_predictions'] = forward_predictions
                
                # Save to intermediate file
                self._save_intermediate(sample_id, intermediate_data)
                
                # Clean up temporary files
                if input_file.exists():
                    input_file.unlink()
                if output_file.exists():
                    output_file.unlink()
                
                return {
                    'status': 'success',
                    'message': 'Successfully generated forward synthesis predictions',
                    'predictions': forward_predictions
                }
                
            except Exception as e:
                error_msg = f"Error reading predictions from output file: {str(e)}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
                
        except Exception as e:
            error_msg = f"Unexpected error in forward synthesis prediction: {str(e)}"
            logging.error(error_msg)
            raise

    async def _wait_for_output(self, output_file: Path, timeout: int = FORWARD_OUTPUT_TIMEOUT) -> bool:
        """Wait for the output file to be generated."""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if output_file.exists():
                return True
            await asyncio.sleep(FORWARD_OUTPUT_CHECK_INTERVAL)
        return False
        

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save data to intermediate file.
        
        Args:
            sample_id: ID of the sample to save
            data: Data to save to intermediate file
        """
        path = self._get_intermediate_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)