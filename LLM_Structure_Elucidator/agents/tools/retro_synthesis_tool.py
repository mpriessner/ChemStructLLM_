"""Tool for generating retrosynthesis predictions using Chemformer."""
from math import log
import os
import logging
import json
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional, List
import pandas as pd
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
import asyncio

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
RETRO_DIR = BASE_DIR / "_temp_folder" / "retro_output"
TEMP_DIR = BASE_DIR / "_temp_folder"

# Constants for Chemformer execution
RETRO_OUTPUT_CHECK_INTERVAL = 5  # seconds
RETRO_OUTPUT_TIMEOUT = 600  # 10 minutes
RETRO_OUTPUT_PATTERN = "retro_predictions.csv"  # Will be formatted with timestamp
RETRO_INPUT_FILENAME = "retro_targets.txt"  # Input filename

SBATCH_SCRIPT = SCRIPTS_DIR / "chemformer_retro_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "chemformer_retro_local.sh"

class RetrosynthesisTool:
    """Tool for generating retrosynthesis predictions using Chemformer."""
    
    def __init__(self):
        """Initialize the Retrosynthesis tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        self.retro_dir = RETRO_DIR
        self.temp_dir = TEMP_DIR
        
        # Create directories if they don't exist
        self.retro_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Add intermediate results directory
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

    async def _prepare_input_from_context(self, context: Dict[str, Any]) -> Path:
        """Prepare input file from context.
        
        Args:
            context: Dictionary containing molecule data and flags
            
        Returns:
            Path to the created input file
        """
        try:
            # Initialize variables
            smiles_list = []
            input_file = self.temp_dir / RETRO_INPUT_FILENAME
            
            # Get molecules from master JSON file
            master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            if master_data_path.exists():
                with open(master_data_path, 'r') as f:
                    master_data = json.load(f)
                # Extract all SMILES from master data
                for molecule_id, molecule_data in master_data.items():
                    if 'smiles' in molecule_data:
                        smiles = self._normalize_smiles(molecule_data['smiles'])
                        if smiles:
                            smiles_list.append(smiles)
                logging.info(f"Extracted {len(smiles_list)} molecules from master JSON")
            else:
                logging.warning("Master JSON file not found")
            
            # If no molecules found in master JSON, try context as fallback
            if not smiles_list and context.get('current_molecule'):
                current_molecule = context['current_molecule']
                if isinstance(current_molecule, dict) and 'SMILES' in current_molecule:
                    smiles = self._normalize_smiles(current_molecule['SMILES'])
                    if smiles:
                        smiles_list.append(smiles)
                elif isinstance(current_molecule, str):
                    smiles = self._normalize_smiles(current_molecule)
                    if smiles:
                        smiles_list.append(smiles)
            
            # Validate SMILES list
            if not smiles_list:
                raise ValueError("No valid SMILES found in master JSON or context")
            
            # Remove duplicates while preserving order
            smiles_list = list(dict.fromkeys(smiles_list))
            
            # Write SMILES to input file
            with open(input_file, 'w') as f:
                for smiles in smiles_list:
                    f.write(f"{smiles}\n")
            
            #logging.info(f"Prepared input file with {len(smiles_list)} molecules")
            return input_file
            
        except Exception as e:
            logging.error(f"Error preparing input file: {str(e)}")
            raise

    def _normalize_smiles(self, smiles: str) -> str:
        """Normalize SMILES string to canonical form."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except ImportError:
            logging.warning("RDKit not available for SMILES normalization")
        return smiles.strip()

    async def _check_starting_materials_exist(self, smiles_list: List[str], intermediate_data: Dict) -> Dict[str, bool]:
        """Check if starting materials already exist in the intermediate data.
        
        Args:
            smiles_list: List of SMILES strings to check
            intermediate_data: The loaded intermediate data containing molecule info and results
            
        Returns:
            Dictionary mapping SMILES to boolean indicating if starting materials exist
        """
        try:
            # Initialize result dictionary
            result = {smiles: False for smiles in smiles_list}
            
            # Check if molecule has starting materials in intermediate
            # if ('step_outputs' in intermediate_data and 
            #     'retrosynthesis' in intermediate_data['step_outputs']):
            #     retro_data = intermediate_data['step_outputs']['retrosynthesis']
            #     if retro_data.get('status') == 'success' and 'predictions' in retro_data:
            #         for smiles in smiles_list:
            #             norm_smiles = self._normalize_smiles(smiles)
            #             # Check if this SMILES has predictions in intermediate
            #             for pred in retro_data['predictions']:
            #                 if self._normalize_smiles(pred.get('target_smiles', '')) == norm_smiles:
            #                     result[smiles] = True
            #                     break
            
            # Also check if starting materials were provided in the original molecule data
            if 'molecule_data' in intermediate_data and 'starting_materials' in intermediate_data['molecule_data']:
                molecule_smiles = intermediate_data['molecule_data'].get('smiles', '')
                if molecule_smiles:
                    norm_molecule_smiles = self._normalize_smiles(molecule_smiles)
                    for smiles in smiles_list:
                        if self._normalize_smiles(smiles) == norm_molecule_smiles:
                            result[smiles] = True
            
            return result
            
        except Exception as e:
            logging.error(f"Error checking starting materials: {str(e)}")
            return {smiles: False for smiles in smiles_list}

    async def _update_master_data(self, predictions_df: pd.DataFrame) -> None:
        """Update the master data file with retrosynthesis predictions.
        
        Args:
            predictions_df: DataFrame containing retrosynthesis predictions with columns:
                          'target_smiles', 'predicted_smiles', 'all_predictions', 'all_log_likelihoods'
        """
        try:
            # Path to master data file
            master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            
            if not master_data_path.exists():
                logging.error("Master data file not found")
                return
            
            # Read current master data
            with open(master_data_path, 'r') as f:
                master_data = json.load(f)

            # logging.info(f"Loaded master data with {len(master_data)} samples")
            # Create mapping of SMILES to predictions
            smiles_to_predictions = {}
            for _, row in predictions_df.iterrows():
                target = self._normalize_smiles(row['target_smiles'])
                
                # Get all predictions and their log likelihoods
                all_preds = [p.strip() for p in row['all_predictions'].split(';') if p.strip()]
                all_logs = [float(l.strip()) for l in row['all_log_likelihoods'].split(';') if l.strip()]
                
                # Create list of (prediction, log_likelihood) pairs
                pred_pairs = list(zip(all_preds, all_logs))
                
                # Sort by log likelihood (highest first) and normalize SMILES
                pred_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Get unique predictions (using normalized SMILES)
                seen = set()
                unique_preds = []
                for pred, _ in pred_pairs:
                    norm_pred = self._normalize_smiles(pred)
                    if norm_pred not in seen:
                        seen.add(norm_pred)
                        unique_preds.append(norm_pred)
                
                # Take top 5 unique predictions
                if unique_preds:
                    smiles_to_predictions[target] = unique_preds[:]

            # logging.info(f"Loaded smiles_to_predictions {smiles_to_predictions}")

            # Update master data
            updated = False
            for sample_id, sample_data in master_data.items():
                if 'smiles' in sample_data:
                    target_smiles = self._normalize_smiles(sample_data['smiles'])
                    if target_smiles in smiles_to_predictions:
                        sample_data['starting_smiles'] = smiles_to_predictions[target_smiles]
                        updated = True
                        #logging.info(f"Updated starting materials for sample {sample_id}")
            # logging.info(f"Loaded master_data {master_data}")
            if updated:
                # Write updated data back to file
                with open(master_data_path, 'w') as f:
                    json.dump(master_data, f, indent=2)
                logging.info("Successfully updated master data with retrosynthesis predictions")
            else:
                logging.warning("No matching SMILES found in master data")
                
        except Exception as e:
            logging.error(f"Error updating master data: {str(e)}")
            raise
 
    async def predict_retrosynthesis(self, molecule_data: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict retrosynthesis for a given molecule.
        
        Args:
            sample_id: ID of the sample to predict retrosynthesis for
            context: Optional context data
            
        Returns:
            Dict containing prediction results
        """
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

            
            # Check if starting materials already exist
            if ('molecule_data' in intermediate_data and 
                'starting_smiles' in intermediate_data['molecule_data'] and
                intermediate_data['molecule_data']['starting_smiles']):
                logging.info(f"Starting materials already exist for sample {sample_id}")
                return {
                    'status': 'success', 
                    'message': 'Starting materials already exist',
                    'predictions': intermediate_data['molecule_data']['starting_smiles']
                }
            
            # Get SMILES from molecule data          
            smiles = intermediate_data['molecule_data'].get('smiles')
            
            # Create input file with single SMILES
            input_file = self.temp_dir / f"{sample_id}_input.txt"
            with open(input_file, 'w') as f:
                f.write(f"{smiles}\n")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.retro_dir / RETRO_OUTPUT_PATTERN.format(timestamp)
            
            logging.info(f"Running retrosynthesis prediction for molecule {smiles}")
            
            # Run prediction locally
            try:
                logging.info("Running retrosynthesis locally")
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
            
            # Wait for output file to be generated
            if not await self._wait_for_output(output_file):
                error_msg = f"Timeout waiting for output file at {output_file}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Read predictions from output file
            try:
                predictions_df = pd.read_csv(output_file)
                
                # Extract starting materials from all_predictions
                starting_materials = []
                
                # Check which format the CSV is in
                if 'all_predictions' in predictions_df.columns:
                    # Original expected format
                    logging.info("Using standard CSV format with all_predictions column")
                    for _, row in predictions_df.iterrows():
                        if 'all_predictions' in row and row['all_predictions']:
                            # Split all predictions by semicolon and add to starting materials
                            all_preds = row['all_predictions'].split(';')
                            starting_materials.extend([p.strip() for p in all_preds if p.strip()])
                elif 'Source' in predictions_df.columns and 'Prediction' in predictions_df.columns:
                    # Alternative format with Source,Prediction columns
                    logging.info("Using alternative CSV format with Source,Prediction columns")
                    for _, row in predictions_df.iterrows():
                        if 'Prediction' in row and row['Prediction']:
                            # Add each prediction as a starting material
                            prediction = row['Prediction'].strip()
                            if prediction:
                                starting_materials.append(prediction)
                else:
                    # Unknown format - try to extract from any columns that might contain SMILES
                    logging.warning(f"Unknown CSV format with columns: {predictions_df.columns}")
                    for col in predictions_df.columns:
                        if col != 'target_smiles' and col != 'Source':  # Skip source columns
                            for val in predictions_df[col]:
                                if isinstance(val, str) and val.strip():
                                    # Basic validation - SMILES usually contain C, N, O, etc.
                                    if any(atom in val for atom in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']):
                                        starting_materials.append(val.strip())
                
                # Remove duplicates while preserving order
                seen = set()
                starting_materials = [x for x in starting_materials if not (x in seen or seen.add(x))]
                
                logging.info(f"Extracted {len(starting_materials)} starting materials from predictions")
                
                # Update intermediate file with starting materials
                intermediate_data['molecule_data']['starting_smiles'] = starting_materials
                self._save_intermediate(sample_id, intermediate_data)
            
                # Return full prediction data for the tool response
                return {
                    'status': 'success',
                    'message': 'Successfully generated retrosynthesis predictions',
                    'predictions': predictions_df.to_dict('records'),
                    'starting_materials': starting_materials
                }
                
            except Exception as e:
                    error_msg = f"Error reading predictions from output file: {str(e)}"
                    logging.error(error_msg)
                    return {'status': 'error', 'message': error_msg}
                
        except Exception as e:
            error_msg = f"Unexpected error in retrosynthesis prediction: {str(e)}"
            logging.error(error_msg)
            raise

    async def _wait_for_output(self, output_file: Path, timeout: int = RETRO_OUTPUT_TIMEOUT) -> bool:
        """Wait for the output file to be generated."""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if output_file.exists():
                return True
            await asyncio.sleep(RETRO_OUTPUT_CHECK_INTERVAL)
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
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
