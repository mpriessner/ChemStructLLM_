"""Tool for generating molecular analogues using Mol2Mol network."""
import os
import logging
import shutil
import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
from models.molecule import MoleculeHandler

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
TEMP_DIR = BASE_DIR / "_temp_folder"
SBATCH_SCRIPT = SCRIPTS_DIR / "mol2mol_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "mol2mol_local.sh"

# Constants for Mol2Mol execution
MOL2MOL_OUTPUT_CHECK_INTERVAL = 5  # seconds
MOL2MOL_OUTPUT_TIMEOUT = 600  # 10 minutes
MOL2MOL_OUTPUT_FILE = "generated_molecules.csv"
MOL2MOL_RUNNING_LOCK = "mol2mol_running.lock"
MOL2MOL_COMPLETE_LOCK = "mol2mol_complete.lock"
MOL2MOL_INPUT_FILENAME = "mol2mol_selection.csv"


class Mol2MolTool:
    """Tool for generating molecular analogues using Mol2Mol network."""
    
    def __init__(self):
        """Initialize the Mol2Mol tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        
        # Create directories if they don't exist
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Validate script existence
        if not SBATCH_SCRIPT.exists():
            raise FileNotFoundError(f"Required SBATCH script not found at {SBATCH_SCRIPT}")
        
        # Validate local script existence
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required LOCAL script not found at {LOCAL_SCRIPT}")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def _prepare_input_file(self, smiles: str, sample_id: str = None) -> Path:
        """Prepare input file for Mol2Mol.
        
        Args:
            smiles: SMILES string of input molecule
            sample_id: Optional sample ID
            
        Returns:
            Path to created input file
        """
        # Validate SMILES
        if not MoleculeHandler.validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Create input DataFrame
        input_data = {
            'SMILES': [smiles],
            'sample-id': [sample_id if sample_id else 'MOL_1']
        }
        df = pd.DataFrame(input_data)
        
        # Save to CSV
        input_file = self.temp_dir / MOL2MOL_INPUT_FILENAME
        await asyncio.to_thread(df.to_csv, input_file, index=False)
        self.logger.info(f"Created input file at: {input_file}")
        
        return input_file

    async def _wait_for_output(self, run_id: str) -> Path:
        """Wait for Mol2Mol generation to complete and return output file path.
        
        Args:
            run_id: Unique identifier for this run (not used anymore, kept for compatibility)
            
        Returns:
            Path to the output file
            
        Raises:
            TimeoutError: If generation doesn't complete within timeout period
        """
        start_time = datetime.now()
        output_file = self.temp_dir / MOL2MOL_OUTPUT_FILE
        running_lock = self.temp_dir / MOL2MOL_RUNNING_LOCK
        complete_lock = self.temp_dir / MOL2MOL_COMPLETE_LOCK
        
        # Create running lock file
        running_lock.touch()
        self.logger.info(f"Created running lock file at: {running_lock}")
        
        while True:
            # Check if complete lock exists and running lock is gone
            if complete_lock.exists() and not running_lock.exists():
                if output_file.exists():
                    try:
                        # Validate output file
                        df = pd.read_csv(output_file)
                        if not df.empty:
                            self.logger.info(f"Found valid output file at: {output_file}")
                            # Clean up lock files
                            if complete_lock.exists():
                                complete_lock.unlink()
                            return output_file
                    except Exception as e:
                        self.logger.warning(f"Found incomplete or invalid output file: {e}")
            
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > MOL2MOL_OUTPUT_TIMEOUT:
                # Clean up lock files
                if running_lock.exists():
                    running_lock.unlink()
                if complete_lock.exists():
                    complete_lock.unlink()
                raise TimeoutError("Timeout waiting for Mol2Mol generation to complete")
            
            # Wait before next check
            await asyncio.sleep(MOL2MOL_OUTPUT_CHECK_INTERVAL)

    # async def _check_existing_predictions(self, molecule_id: str) -> Optional[List[str]]:
    #     """Check if mol2mol predictions already exist for a given molecule ID.
        
    #     Args:
    #         molecule_id: ID of the molecule to check
            
    #     Returns:
    #         List of predicted SMILES if they exist, None otherwise
    #     """
    #     try:
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if not master_data_path.exists():
    #             return None
            
    #         # Read master data
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         # Check if molecule exists and has predictions
    #         if molecule_id in master_data and 'mol2mol_predictions' in master_data[molecule_id]:
    #             predictions = master_data[molecule_id]['mol2mol_predictions']
    #             if predictions:  # Only return if there are actual predictions
    #                 return predictions
    #         return None
            
    #     except Exception as e:
    #         self.logger.error(f"Error checking existing predictions: {str(e)}")
    #         return None

    async def generate_analogues(self, smiles: str, sample_id: str = None) -> Dict[str, Any]:
        """Generate analogues for a given SMILES string.
        
        Args:
            smiles: Input SMILES string
            sample_id: Optional sample ID
            
        Returns:
            Dict containing status and results/error message
        """
        self.logger.warning("mol2mol_tool.generate_analogues execute")

        try:
            if not sample_id:
                raise ValueError("No sample_id provided")

            # Load or create intermediate data
            context = {'smiles': smiles, 'sample_id': sample_id}
            intermediate_data = self._load_or_create_intermediate(sample_id, context)
            
            # Check if predictions already exist
            if ('molecule_data' in intermediate_data and 
                'mol2mol_results' in intermediate_data['molecule_data'] and
                intermediate_data['molecule_data']['mol2mol_results']['status'] == 'success'):
                self.logger.info(f"Found existing mol2mol predictions for sample {sample_id}")
                return {
                    'status': 'success',
                    'message': 'Mol2mol predictions already exist',
                    'predictions': next(iter(intermediate_data['molecule_data']['mol2mol_results']['generated_analogues_target'].values()))
                }

            # Create unique output filename
            output_filename = f"generated_molecules_{sample_id}.csv"
            output_file = self.temp_dir / output_filename
            
            # Create input file
            input_file = await self._prepare_input_file(smiles, sample_id)
            
            # Check if using SLURM
            use_slurm = False  # Default to local execution
            
            # Check CUDA availability for local execution
            try:
                import torch
                if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                    self.logger.warning("CUDA not available. Switching to SLURM execution.")
                    use_slurm = True
            except ImportError:
                self.logger.warning("PyTorch not found. Switching to SLURM execution.")
                use_slurm = True
            
            if use_slurm:
                # Submit SLURM job
                self.logger.info("Submitting Mol2Mol SLURM job")
                try:
                    process = await asyncio.create_subprocess_exec(
                        'sbatch',
                        str(SBATCH_SCRIPT),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        return {
                            'status': 'error',
                            'message': f'SLURM submission failed: {stderr.decode()}'
                        }
                        
                    # Extract job ID from sbatch output
                    job_id = stdout.decode().strip().split()[-1]
                    self.logger.info(f"SLURM job submitted with ID: {job_id}")
                    
                    # Wait for job completion
                    output_file = await self._wait_for_output(job_id)
                    
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'SLURM execution failed: {str(e)}'
                    }
            else:
                # Execute locally
                self.logger.info("Running Mol2Mol locally")
                LOCAL_SCRIPT.chmod(0o755)  # Make script executable
                try:
                    process = await asyncio.create_subprocess_exec(
                        str(LOCAL_SCRIPT),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()  # Wait for process to finish
                    
                    if process.returncode != 0:
                        return {
                            'status': 'error',
                            'message': f'Local execution failed: {stderr.decode()}'
                        }
                    
                    # Process finished, check output file
                    temp_output = self.temp_dir / MOL2MOL_OUTPUT_FILE
                    if not temp_output.exists():
                        return {
                            'status': 'error',
                            'message': f'Output file not found at {temp_output}'
                        }
                    
                    # If using unique filename, rename the output file
                    if output_filename != MOL2MOL_OUTPUT_FILE:
                        temp_output.rename(output_file)
                    
                    # Validate output file
                    try:
                        df = pd.read_csv(output_file)
                        if df.empty:
                            return {
                                'status': 'error',
                                'message': 'Generated file is empty'
                            }
                    except Exception as e:
                        return {
                            'status': 'error',
                            'message': f'Failed to read output file: {str(e)}'
                        }
                        
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Local execution failed: {str(e)}'
                    }
            
            # Read predictions and store in intermediate file
            predictions_df = pd.read_csv(output_file)
            # Extract predictions - first row contains input SMILES, subsequent rows are predictions
            sample_id_col = next(iter(predictions_df.columns))  # Get first column name (sample ID)
            predictions = predictions_df[sample_id_col][1:].tolist()  # Skip first row (input SMILES)
            self.logger.info(f"Generated predictions for sample {sample_id}: {predictions}")
            # self.logger.info(f"Intermediate data for sample {sample_id}: {json.dumps(intermediate_data, indent=2)}")
            
            # Store predictions in intermediate data under molecule_data
            if 'molecule_data' not in intermediate_data:
                intermediate_data['molecule_data'] = {}
                
            intermediate_data['molecule_data']['mol2mol_results'] = {
                'generated_analogues_target': {
                    smiles: predictions  # Target SMILES -> list of generated analogues
                },
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            self._save_intermediate(sample_id, intermediate_data)
            
            # Clean up temporary files
            if input_file.exists():
                input_file.unlink()
            if output_file.exists():
                output_file.unlink()
            
            return {
                'status': 'success',
                'message': 'Successfully generated analogues',
                'predictions': predictions
            }
            
        except Exception as e:
            error_msg = f"Error in mol2mol generation: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}


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
            sample_id: ID of the sample to save data for
            data: Dictionary containing data to save
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    # async def _update_master_data(self, analogues_file: Path, molecule_id: str) -> None:
    #     """Update the master data file with generated molecular analogues.
        
    #     Args:
    #         analogues_file: Path to the file containing generated analogues
    #         molecule_id: ID of the molecule to update
            
    #     Raises:
    #         FileNotFoundError: If master data file doesn't exist
    #         ValueError: If analogues file is empty or invalid
    #     """
    #     try:
    #         self.logger.info(f"Starting master data update for molecule {molecule_id}")
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            
    #         if not master_data_path.exists():
    #             error_msg = f"Master data file not found at {master_data_path}. Please upload a CSV file first."
    #             self.logger.error(error_msg)
    #             raise FileNotFoundError(error_msg)
                
    #         self.logger.info(f"Reading existing master data from {master_data_path}")
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         self.logger.info(f"Reading generated analogues from {analogues_file}")
    #         # Read generated analogues directly from file
    #         with open(analogues_file, 'r') as f:
    #             # Skip empty lines and strip whitespace
    #             generated_smiles = [line.strip() for line in f if line.strip()]
                
    #         self.logger.info(f"Found {len(generated_smiles)} generated SMILES")
            
    #         # Initialize molecule entry if it doesn't exist
    #         if molecule_id not in master_data:
    #             master_data[molecule_id] = {}
            
    #         # Store the predictions
    #         master_data[molecule_id]['mol2mol_predictions'] = generated_smiles
    #         self.logger.info(f"Added {len(generated_smiles)} predictions for molecule {molecule_id}")
            
    #         # Write updated data back to file
    #         self.logger.info("Writing updated master data")
    #         with open(master_data_path, 'w') as f:
    #             json.dump(master_data, f, indent=2)
    #         self.logger.info("Successfully updated master data file")
            
    #         # Delete the generated molecules file to prevent accidental reuse
    #         if analogues_file.exists():
    #             analogues_file.unlink()
    #             self.logger.info(f"Deleted generated molecules file: {analogues_file}")
                
    #     except Exception as e:
    #         self.logger.error(f"Error updating master data: {str(e)}", exc_info=True)
    #         raise

    # async def process_batch(self, molecules: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    #     """Process a batch of molecules.
        
    #     Args:
    #         molecules: List of dictionaries containing 'SMILES' and optionally 'sample_id'
            
    #     Returns:
    #         List of generation results for each molecule
    #     """
    #     results = []
    #     for mol in molecules:
    #         try:
    #             # Extract data
    #             smiles = mol['SMILES']
    #             sample_id = mol.get('sample_id')
                
    #             # Generate analogues
    #             result = await self.generate_analogues(smiles, sample_id)
    #             results.append(result)
                
    #         except Exception as e:
    #             results.append({
    #                 'status': 'error',
    #                 'message': f'Failed to process molecule: {str(e)}',
    #                 'smiles': mol.get('SMILES', 'unknown')
    #             })
                
    #     return results

    # async def process_all_molecules(self) -> Dict[str, Any]:
    #     """Process all molecules in the master data file for mol2mol predictions.
        
    #     Returns:
    #         Dictionary containing processing results
    #     """
    #     try:
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if not master_data_path.exists():
    #             return {'status': 'error', 'message': 'Master data file not found'}
            
    #         # Read master data
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         results = []
    #         # Process each molecule
    #         for molecule_id, molecule_data in master_data.items():
    #             if 'smiles' in molecule_data:
    #                 # Prepare molecule data for generation
    #                 mol_input = {
    #                     'SMILES': molecule_data['smiles'],
    #                     'sample_id': molecule_id,
    #                     'name': molecule_id
    #                 }
                    
    #                 # Generate analogues with unique filename
    #                 generation_result = await self.generate_analogues(mol_input['SMILES'], mol_input['sample_id'])
                    
    #                 if generation_result['status'] == 'success':
    #                     # Update master data with predictions using the output file path from generation
    #                     output_file = Path(generation_result['output_file'])
    #                     await self._update_master_data(output_file, molecule_id)
    #                     results.append({
    #                         'molecule_id': molecule_id,
    #                         'status': 'success'
    #                     })
    #                 else:
    #                     results.append({
    #                         'molecule_id': molecule_id,
    #                         'status': 'error',
    #                         'message': generation_result['message']
    #                     })
            
    #         return {
    #             'status': 'success',
    #             'message': f'Processed {len(results)} molecules',
    #             'results': results
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Error processing molecules: {str(e)}")
    #         return {
    #             'status': 'error',
    #             'message': str(e)
    #         }
