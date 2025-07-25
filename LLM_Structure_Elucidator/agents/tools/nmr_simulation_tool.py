"""Tool for simulating NMR spectra from molecular structures using SGNN."""
import os
import logging
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys
import ast
import subprocess
import json
import time
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
import asyncio
import os   

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This will force the configuration even if logging was already configured
)
logger = logging.getLogger(__name__)

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
SIMULATIONS_DIR = BASE_DIR / "_temp_folder"
INTERMEDIATE_DIR = SIMULATIONS_DIR / "intermediate_results"
SGNN_DATA_DIR = BASE_DIR / "_temp_folder"
SBATCH_SCRIPT = SCRIPTS_DIR / "sgnn_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "sgnn_local.sh"
SGNN_INPUT_FILENAME = "current_molecule.csv"

# Constants for SGNN output processing
SGNN_OUTPUT_TYPES = ['1H', '13C', 'COSY', 'HSQC']
SGNN_OUTPUT_CHECK_INTERVAL = 5  # seconds
SGNN_OUTPUT_TIMEOUT = 300  # 5 minutes to wait for output files
SGNN_OUTPUT_PATTERN = "nmr_prediction_{type}.csv"  # e.g., nmr_prediction_1H.csv
SGNN_TIMEOUT = 300  # 5 minutes to wait for output files

class NMRSimulationTool:
    """Tool for simulating NMR spectra from molecular structures using SGNN."""
    
    def __init__(self):
        """Initialize the NMR simulation tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        self.simulations_dir = SIMULATIONS_DIR
        self.sgnn_data_dir = SGNN_DATA_DIR
        self.intermediate_dir = INTERMEDIATE_DIR
        
        # Create directories if they don't exist
        self.simulations_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        self.intermediate_dir.mkdir(exist_ok=True, parents=True)  # parents=True to create parent dirs if needed
        
        # Validate script existence
        if not SBATCH_SCRIPT.exists():
            raise FileNotFoundError(f"Required SBATCH script not found at {SBATCH_SCRIPT}")
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required local script not found at {LOCAL_SCRIPT}")
            
        # Validate environment
        try:
            import torch
            if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                self.logger.warning("CUDA not available for local execution. SLURM execution will be forced.")
        except ImportError:
            self.logger.warning("PyTorch not found. Please ensure the SGNN environment is activated.")
        
        # Set up logger
        self.logger = logging.getLogger(__name__)

    async def _wait_for_sgnn_outputs(self, smiles: str) -> Dict[str, Path]:
        """Wait for SGNN outputs to be generated.
        
        Args:
            smiles: SMILES string of molecule
            
        Returns:
            Dictionary mapping NMR type to output file path
        """
        output_files = {}
        start_time = time.time()
        

        # Wait for new output files
        while True:
            # Check each type of NMR output
            for nmr_type in SGNN_OUTPUT_TYPES:
                output_file = self.sgnn_data_dir / SGNN_OUTPUT_PATTERN.format(type=nmr_type)
                if output_file.exists() and output_file not in output_files.values():
                    self.logger.info(f"Found valid output for {nmr_type} NMR")
                    output_files[nmr_type] = output_file
            
            # Check if we have all outputs
            if len(output_files) == len(SGNN_OUTPUT_TYPES):
                break
                
            # Check timeout
            if time.time() - start_time > SGNN_TIMEOUT:
                raise TimeoutError("Timeout waiting for SGNN outputs")
                
            time.sleep(0.1)
            
        return output_files

    async def _prepare_input_data(self, molecule_data: str, simulation_mode: str) -> pd.DataFrame:
        """Prepare input data for NMR simulation from master JSON file.
        
        Args:
            molecule_data: Path to master JSON file
            simulation_mode: Always 'batch' mode for efficiency
            
        Returns:
            DataFrame with SMILES and sample-id columns
        """
        if not os.path.exists(molecule_data):
            raise FileNotFoundError(f"Master JSON file not found: {molecule_data}")
            
        self.logger.info(f"Loading molecular data from master JSON: {molecule_data}")
        
        try:
            # Read master JSON file
            with open(molecule_data, 'r') as f:
                master_data = json.load(f)
                
            # Extract SMILES and sample IDs
            simulation_input = []
            for sample_id, data in master_data.items():
                if 'smiles' in data:  # Make sure we have SMILES data
                    simulation_input.append({
                        'SMILES': data['smiles'],
                        'sample-id': sample_id  # Use the exact sample ID from master data
                    })
                    
            if not simulation_input:
                raise ValueError("No valid molecules found in master JSON file")
                
            df = pd.DataFrame(simulation_input)
            self.logger.info(f"Prepared simulation input for {len(df)} molecules")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to prepare simulation input from JSON: {str(e)}")
            raise

    async def simulate_batch(self, master_data_path: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate NMR spectrum for molecules in master JSON file.
        
        Args:
            master_data_path: Path to master JSON file containing molecular data
            context: Optional context including:
                - use_slurm: Whether to use SLURM (default: False)
            
        Returns:
            Dictionary containing simulation results
        """
        try:
            self.logger.info("Starting NMR simulation process")
            context = context or {}
                

                    
            # Always use batch mode
            simulation_mode = 'batch'
            
            # Check CUDA availability for local execution
            use_slurm = context.get('use_slurm', False)
            if not use_slurm:
                try:
                    import torch
                    if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                        self.logger.warning("CUDA not available. Switching to SLURM execution.")
                        use_slurm = True
                except ImportError:
                    self.logger.warning("PyTorch not found. Switching to SLURM execution.")
                    use_slurm = True
            
            # Prepare input data from master JSON
            try:
                df = await self._prepare_input_data(master_data_path, simulation_mode)
                self.logger.info(f"Prepared input data with {len(df)} molecules from master JSON")
            except Exception as e:
                raise ValueError(f"Failed to prepare input data from master JSON: {str(e)}")
            
            try:
                self.logger.info("Input validation")
                
                # Ensure temp directory exists
                self.sgnn_data_dir.mkdir(exist_ok=True)
                
                # Define target path for SGNN input
                sgnn_input_path = self.sgnn_data_dir / SGNN_INPUT_FILENAME
                
                # Copy input file to SGNN location with fixed name, replacing if exists
                self.logger.info(f"Copying input file to SGNN location: {sgnn_input_path}")
                if sgnn_input_path.exists():
                    self.logger.info(f"Removing existing file at {sgnn_input_path}")
                    sgnn_input_path.unlink()  # Explicitly remove existing file
                df.to_csv(sgnn_input_path, index=False)

                # Validate file content
                try:
                    df = pd.read_csv(sgnn_input_path)
                    if df.empty:
                        raise ValueError("Input CSV file is empty")
                except Exception as e:
                    raise ValueError(f"Invalid CSV file format: {str(e)}")
                
                self.logger.info("Input file processed successfully")
                
                # Create timestamp for unique output directory
                timestamp = datetime.now().strftime("%H%M%S")
                output_dir = self.sgnn_data_dir / f"nmr_output_{timestamp}"
                output_dir.mkdir(exist_ok=True)
                
                # Store original filename for later use (without extension)
                self.current_molecule_name = sgnn_input_path.stem
                
                # Determine execution mode
                if use_slurm:
                    # Execute using SLURM
                    self.logger.info("Running NMR simulation with SLURM")
                    # Pass arguments individually to execute_sbatch
                    self.logger.info(f"Running SLURM with input file: {sgnn_input_path}")
                    job_id = await execute_sbatch(str(SBATCH_SCRIPT), "--input_file", str(sgnn_input_path))
                    success = await wait_for_job_completion(job_id)
                    if not success:
                        self.logger.error("SLURM job failed")
                        return {
                            'status': 'error',
                            'message': 'SLURM job failed'
                        }
                else:
                    # Execute locally
                    self.logger.info("Running NMR simulation locally")
                    cmd = [str(LOCAL_SCRIPT), "--input_file", str(sgnn_input_path)]
                    self.logger.info(f"Running command: {' '.join(cmd)}")
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Local execution failed with error: {str(e)}")
                        return {
                            'status': 'error',
                            'message': f'Local execution failed: {str(e)}'
                        }
                
                self.logger.info("NMR simulation completed successfully")
                
                # Wait for and validate output files
                try:
                    output_files = await self._wait_for_sgnn_outputs(df['SMILES'].iloc[0])
                    self.logger.info(f"Found all required output files: {list(output_files.keys())}")
                    
                    # Compile results into single file
                    result_path = await self._compile_results(output_files)
                    
                    # Update master data with simulation results
                    # await self._update_master_data(result_path)

                    # Clean up temporary directories
                    try:
                        # Remove timestamp-based output directory
                        if output_dir.exists():
                            shutil.rmtree(output_dir)
                            self.logger.info(f"Cleaned up temporary output directory: {output_dir}")
                    except Exception as e:
                        self.logger.warning(f"Error during cleanup: {str(e)}")

                    return {
                        "status": "success",
                        "type": "nmr_prediction",
                        "data": {
                            "message": "NMR simulation completed and results compiled",
                            "result_file": str(result_path),
                            "output_files": {k: str(v) for k, v in output_files.items()}
                        }
                    }
                    
                except TimeoutError:
                    return {
                        'status': 'error',
                        'message': 'Timeout waiting for output files'
                    }
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Failed to process output files: {str(e)}'
                    }
            
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'NMR simulation failed: {str(e)}'
                }
                
        except Exception as e:
            self.logger.error(f"Error in NMR simulation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def _compile_results(self, output_files: Dict[str, Path]) -> Path:
        try:
            input_file = self.sgnn_data_dir / SGNN_INPUT_FILENAME
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
                
            # Read and validate input file
            input_df = pd.read_csv(input_file)
            if input_df.empty:
                raise pd.errors.EmptyDataError("Input file is empty")

            # Read and validate each NMR prediction file
            for nmr_type, file_path in output_files.items():
                if not file_path.exists():
                    self.logger.warning(f"NMR prediction file not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path)
                if df.empty:
                    self.logger.warning(f"NMR prediction file is empty: {file_path}")
                    continue
                    
                if 'shifts' not in df.columns or 'sample-id' not in df.columns:
                    self.logger.warning(f"Missing required columns in {nmr_type} NMR predictions")
                    continue
                    
                # Create a dictionary to store processed shifts by sample ID
                processed_shifts_dict = {}
                
                # Process each row
                for _, row in df.iterrows():
                    sample_id = row['sample-id']
                    shift_string = row['shifts']
                    
                    try:
                        # Convert string representation to actual list using ast.literal_eval
                        shift_list = ast.literal_eval(shift_string)
                        
                        # Handle different NMR types
                        if nmr_type == '1H':
                            # 1H NMR format: keep both shift and intensity as tuples
                            shift_values = [(float(tup[0]), float(tup[1])) for tup in shift_list]
                        elif nmr_type in ['COSY', 'HSQC']:
                            # COSY and HSQC format: keep correlation pairs as tuples
                            shift_values = [(float(tup[0]), float(tup[1])) for tup in shift_list]
                        else:
                            # 13C NMR format: direct list of float values
                            shift_values = [float(val) for val in shift_list]
                            
                        processed_shifts_dict[sample_id] = shift_values
                    except Exception as e:
                        self.logger.warning(f"Error processing shift value for {sample_id}: {str(e)}")
                        processed_shifts_dict[sample_id] = []

                # Add to input DataFrame with correct column name, using empty list for missing predictions
                column_name = f"{nmr_type}_NMR_sim" if nmr_type in ['1H', '13C'] else f"{nmr_type}_sim"
                # Create a new Series with the same index as input_df
                shifts_series = pd.Series(index=input_df.index, dtype=object)
                for idx, sample_id in enumerate(input_df['sample-id']):
                    shifts_series[idx] = processed_shifts_dict.get(sample_id, [])
                input_df[column_name] = shifts_series
                
            # Create output path in simulations directory
            self.simulations_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.simulations_dir / f"{self.current_molecule_name}_sim.csv"
        
            # Save combined results
            input_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved compiled results to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error compiling results: {str(e)}")
            raise ValueError(f"Failed to compile NMR prediction results: {str(e)}")

    # async def _update_master_data(self, simulation_results_file: Path) -> None:
    #     """Update the master data file with NMR simulation results.
        
    #     Args:
    #         simulation_results_file: Path to the CSV containing NMR simulation results
            
    #     The CSV file contains:
    #     - sample-id: Used to identify the molecule in the master JSON
    #     - NMR simulation columns (1H_NMR_sim, 13C_NMR_sim, COSY_sim, HSQC_sim)
            
    #     Raises:
    #         FileNotFoundError: If master data file doesn't exist
    #         ValueError: If simulation results file is empty or invalid
    #     """
    #     try:
    #         self.logger.info("Starting master data update from simulation results")
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            
    #         if not master_data_path.exists():
    #             error_msg = f"Master data file not found at {master_data_path}"
    #             self.logger.error(error_msg)
    #             raise FileNotFoundError(error_msg)
                
    #         if not simulation_results_file.exists():
    #             error_msg = f"Simulation results file not found at {simulation_results_file}"
    #             self.logger.error(error_msg)
    #             raise FileNotFoundError(error_msg)
                
    #         self.logger.info(f"Reading existing master data from {master_data_path}")
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         self.logger.info(f"Reading NMR simulation results from {simulation_results_file}")
    #         simulation_df = pd.read_csv(simulation_results_file)
            
    #         if simulation_df.empty:
    #             raise ValueError("Simulation results file is empty")
                
    #         # NMR column mapping from CSV to JSON
    #         nmr_mapping = {
    #             '1H_NMR_sim': '1H_sim',
    #             '13C_NMR_sim': '13C_sim',
    #             'COSY_sim': 'COSY_sim',
    #             'HSQC_sim': 'HSQC_sim'
    #         }
            
    #         # Process each row in the simulation results
    #         for idx, row in simulation_df.iterrows():
    #             sample_id = row['sample-id']
    #             self.logger.info(f"Processing NMR data for sample: {sample_id}")
                
    #             # Create entry if it doesn't exist
    #             if sample_id not in master_data:
    #                 master_data[sample_id] = {'nmr_data': {}}
    #             elif 'nmr_data' not in master_data[sample_id]:
    #                 master_data[sample_id]['nmr_data'] = {}
                
    #             # Process each NMR type for this sample
    #             for csv_col, json_key in nmr_mapping.items():
    #                 if csv_col in simulation_df.columns:
    #                     try:
    #                         prediction_data = row[csv_col]
    #                         # Handle empty or missing predictions
    #                         if pd.isna(prediction_data):
    #                             master_data[sample_id]['nmr_data'][json_key] = []
    #                             self.logger.warning(f"No {json_key} predictions for sample {sample_id}")
    #                         else:
    #                             # Convert string representation to list if needed
    #                             if isinstance(prediction_data, str):
    #                                 prediction_data = ast.literal_eval(prediction_data)
    #                             master_data[sample_id]['nmr_data'][json_key] = prediction_data
    #                             self.logger.info(f"Added {json_key} predictions for sample {sample_id}")
    #                     except Exception as e:
    #                         self.logger.warning(f"Failed to process {csv_col} for sample {sample_id}: {str(e)}")
    #                         master_data[sample_id]['nmr_data'][json_key] = []
            
    #         # Write updated data back to file
    #         self.logger.info("Writing updated master data")
    #         with open(master_data_path, 'w') as f:
    #             json.dump(master_data, f, indent=2)
    #         self.logger.info("Successfully updated master data file")
            
    #         # Clean up simulation results file
    #         # if simulation_results_file.exists():
    #         #     simulation_results_file.unlink()
    #         #     self.logger.info(f"Deleted simulation results file: {simulation_results_file}")
                
    #     except Exception as e:
    #         self.logger.error(f"Error updating master data: {str(e)}", exc_info=True)
    #         raise

    async def simulate_nmr(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Run NMR simulation for a molecule.
        
        Args:
            sample_id: ID of the sample to simulate
            context: Optional context data if not loading from master file
            
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info(f"Selected sample_id: {sample_id}")

        # Delete any existing output files first
        for nmr_type in SGNN_OUTPUT_TYPES:
            output_file = self.sgnn_data_dir / SGNN_OUTPUT_PATTERN.format(type=nmr_type)
            if output_file.exists():
                self.logger.info(f"Removing existing output file for {nmr_type} NMR")
                output_file.unlink()

        # Load or create intermediate file
        intermediate_data = self._load_or_create_intermediate(sample_id, context)
        
        # Get SMILES from molecule data
        smiles = intermediate_data['molecule_data'].get('smiles')
        if not smiles:
            raise ValueError(f"No SMILES found for sample {sample_id}")
            
        # Canonicalize SMILES for consistent comparison
        smiles = self._canonicalize_smiles(smiles)
        
        # Check if we have all required NMR data
        self.logger.info(f"Checking NMR data: {intermediate_data['molecule_data'].get('nmr_data', {})}")
        nmr_data = intermediate_data['molecule_data'].get('nmr_data', {})
        
        # Check for simulation results (not experimental data)
        required_sims = ['1H_exp', '13C_exp', 'COSY_exp', 'HSQC_exp', '1H_sim', '13C_sim', 'COSY_sim', 'HSQC_sim']
        existing_sims = [key for key in required_sims if key in nmr_data]
        self.logger.info(f"Required simulations: {required_sims}")
        self.logger.info(f"Found simulations: {existing_sims}")
        
        if len(existing_sims) == len(required_sims):
            self.logger.info(f"All NMR simulations exist for sample {sample_id}")
            return {
                'status': 'success',
                'message': 'NMR simulations already exist',
                'predictions': nmr_data
            }
        else:
            missing_sims = set(required_sims) - set(existing_sims)
            self.logger.info(f"Missing simulations: {missing_sims}, will run simulation")
        
        # Create input file
        input_file = self.simulations_dir / SGNN_INPUT_FILENAME
        df = pd.DataFrame([{'SMILES': smiles, "sample-id": sample_id}])
        df.to_csv(input_file, index=False)
        
        # Run prediction locally
        try:
            self.logger.info("Running NMR simulation locally")
            self.logger.info(f"Using script: {LOCAL_SCRIPT}")
            self.logger.info(f"Input file: {input_file}")
            
            process = await asyncio.create_subprocess_exec(
                str(LOCAL_SCRIPT),
                f"--input_file={input_file}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Capture output in real-time
            stdout, stderr = await process.communicate()
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            # Log all output
            if stdout_str:
                self.logger.info(f"SGNN stdout:\n{stdout_str}")
            if stderr_str:
                self.logger.error(f"SGNN stderr:\n{stderr_str}")
            
            if process.returncode != 0:
                error_msg = f"Local execution failed with return code {process.returncode}"
                if stderr_str:
                    error_msg += f": {stderr_str}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
                
            self.logger.info("SGNN script completed successfully")
            
        except Exception as e:
            error_msg = f"Error during local execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {'status': 'error', 'message': error_msg}
        
        # Wait for output files to be generated
        try:
            self.logger.info("Waiting for output files...")
            output_files = await self._wait_for_sgnn_outputs(smiles)
            self.logger.info(f"Found output files: {list(output_files.keys())}")
        except TimeoutError as e:
            error_msg = str(e)
            self.logger.error(error_msg)
            # Check directory contents one last time
            if self.sgnn_data_dir.exists():
                self.logger.error(f"Final contents of output directory {self.sgnn_data_dir}:")
                for f in self.sgnn_data_dir.iterdir():
                    self.logger.error(f"  {f.name} ({f.stat().st_size} bytes)")
            return {'status': 'error', 'message': error_msg}
        
        # Process results and update intermediate file
        nmr_data = {}
        
        for nmr_type, output_file in output_files.items():
            try:
                df = pd.read_csv(output_file)
                self.logger.info(f"Processing {nmr_type} NMR data")
                self.logger.info(f"DataFrame columns: {df.columns.tolist()}")
                self.logger.info(f"DataFrame content:\n{df}")
                
                # Filter for current sample's data
                if 'SMILES' in df.columns:
                    current_sample_data = df[df['SMILES'] == smiles]
                    if current_sample_data.empty:
                        self.logger.error(f"No data found for SMILES {smiles} in {nmr_type} output")
                        continue
                else:
                    current_sample_data = df
                    
                self.logger.info(f"Filtered data for current sample:\n{current_sample_data}")
                
                if nmr_type == '1H':
                    # Convert string representation of shifts to actual list of [shift, intensity] pairs
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert tuples to lists and ensure they're floats
                    shifts_list = [[float(shift), float(intensity)] for shift, intensity in shifts_list]
                    nmr_data['1H_sim'] = shifts_list
                    
                elif nmr_type == '13C':
                    # Convert string representation of shifts to list of floats
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert to list of floats
                    shifts_list = [float(shift) for shift in shifts_list]
                    nmr_data['13C_sim'] = shifts_list
                    
                elif nmr_type == 'COSY':
                    # Convert string representation of shifts to list of [shift1, shift2] pairs
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert tuples to lists and ensure they're floats
                    shifts_list = [[float(shift1), float(shift2)] for shift1, shift2 in shifts_list]
                    nmr_data['COSY_sim'] = shifts_list
                    
                elif nmr_type == 'HSQC':
                    # Convert string representation of shifts to list of [1H_shift, 13C_shift] pairs
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert to list of lists and ensure they're floats
                    shifts_list = [[float(h_shift), float(c_shift)] for h_shift, c_shift in shifts_list]
                    nmr_data['HSQC_sim'] = shifts_list
                    
            except Exception as e:
                self.logger.error(f"Error processing {nmr_type} NMR data: {str(e)}")
                self.logger.error(f"Error details:", exc_info=True)
                continue
         
        if not nmr_data:
            raise ValueError("Failed to process any NMR simulation data")
        
        # Update intermediate file with NMR data
        intermediate_data['molecule_data']['nmr_data'].update(nmr_data)
        self._save_intermediate(sample_id, intermediate_data)
        
        return {
            'status': 'success',
            'message': 'Successfully simulated NMR spectra',
            'predictions': nmr_data
        }
        
        
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

    def _save_intermediate(self, sample_id: str, data: Dict) -> None:
        """Save data to intermediate file."""
        intermediate_path = self._get_intermediate_path(sample_id)
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _canonicalize_smiles(self, smiles: str) -> str:
        """Convert SMILES to canonical form for consistent comparison.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES string
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Could not parse SMILES: {smiles}")
                return smiles
            return Chem.MolToSmiles(mol, canonical=True)
        except ImportError:
            self.logger.warning("RDKit not available, using raw SMILES")
            return smiles
