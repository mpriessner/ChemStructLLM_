"""Tool for predicting molecular structures using Multi-Modal Spectral Transformer."""
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
import pickle
import uuid
from rdkit import Chem  # Add RDKit for SMILES canonicalization
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
from models.molecule import MoleculeHandler

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
TEMP_DIR = BASE_DIR / "_temp_folder" / "mmst_temp"
SBATCH_SCRIPT = SCRIPTS_DIR / "mmst_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "mmst_local.sh"

# Constants for MMST execution
MMST_OUTPUT_CHECK_INTERVAL = 5  # seconds
MMST_OUTPUT_TIMEOUT = 1800  # 30 minutes (longer timeout due to fine-tuning)
MMST_SUMMARY_FILE = 'mmst_final_results.json'  # New summary file name
MMST_INPUT_FILENAME = "mmst_input.csv"

MMST_SGNN_OUTPUT_DIR = TEMP_DIR / "sgnn_output"

class MMSTTool:
    """Tool for predicting molecular structures using Multi-Modal Spectral Transformer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MMST tool."""
        self.config_path = config_path
        self.temp_dir = TEMP_DIR
        self.start_time = time.time()
        self.intermediate_dir = BASE_DIR / "_temp_folder" / "intermediate_results"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp-based run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = TEMP_DIR / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created run directory: {self.run_dir}")

    def get_run_dir(self) -> Path:
        """Get the current run directory."""
        return self.run_dir

    def _get_work_dir(self, sample_id: Optional[str] = None) -> Path:
        """Get working directory for a sample within the current run directory."""
        return self.run_dir / (sample_id if sample_id else 'MOL_1')

    def get_sample_dirs(self, sample_id: str) -> dict:
        """Create and return dictionary of sample-specific directories within run directory.
        
        Args:
            sample_id: Sample ID string
            
        Returns:
            Dictionary containing paths for each subdirectory
        """
        sample_dir = self._get_work_dir(sample_id)
        dirs = {
            'sample': sample_dir,
            'models': sample_dir / 'models',
            'sgnn_output': sample_dir / 'sgnn_output',
            'experimental_data': sample_dir / 'experimental_data',
            'test_results': sample_dir / 'test_results'
        }
        
        # Create all directories
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return dirs

    async def _prepare_input_file(self, smiles: str, molecule_id: str = None) -> Path:
        """Prepare input file for MMST.
        
        Args:
            smiles: SMILES string of reference molecule
            molecule_id: Optional molecule identifier
            
        Returns:
            Path to created input file
        """
        # Validate SMILES
        if not MoleculeHandler.validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Prepare input data
        input_data = {
            'SMILES': [smiles],
            'sample-id': [molecule_id if molecule_id else 'MOL_1']
        }
        df = pd.DataFrame(input_data)
        
        # Create sample-specific directory
        sample_dir = self._get_work_dir(molecule_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV in sample directory
        input_file = sample_dir / MMST_INPUT_FILENAME
        await asyncio.to_thread(df.to_csv, input_file, index=False)
        self.logger.info(f"Created input file at: {input_file}")
        
        return input_file

    async def _wait_for_output(self, sample_id: Optional[str] = None) -> Optional[Path]:
        """Wait for MMST output file to be generated and return its path."""
        # Use sample directory if provided
        work_dir = self._get_work_dir(sample_id)
        test_results_dir = work_dir / "test_results"  # Add test_results subdirectory
        output_file = test_results_dir / MMST_SUMMARY_FILE
        
        while True:
            # Check if output file exists and is valid
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        mmst_results = json.load(f)
                    
                    # Validate the new JSON structure
                    if 'runs' in mmst_results and isinstance(mmst_results['runs'], list):
                        # Check if we have at least one run with required data
                        for run in mmst_results['runs']:
                            required_keys = ['exp_results_file', 'final_performance', 'model_save_path']
                            if all(key in run for key in required_keys):
                                # Verify that exp_results_file exists
                                exp_results_file = Path(run['exp_results_file'])
                                if exp_results_file.exists():
                                    return output_file  # Return the Path object
                                else:
                                    self.logger.warning(f"Experimental results file not found: {exp_results_file}")
                        
                        # If we get here, no valid run was found
                        self.logger.warning("No valid runs found in results file")
                    else:
                        self.logger.warning("Invalid results format: 'runs' array not found")
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in {output_file}")
                except Exception as e:
                    self.logger.warning(f"Error processing results file: {str(e)}")
            
            # Check timeout
            if time.time() - self.start_time > MMST_OUTPUT_TIMEOUT:
                self.logger.error("Timeout waiting for MMST output")
                return None
            
            time.sleep(MMST_OUTPUT_CHECK_INTERVAL)

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
        """Save intermediate data to file.
        
        Args:
            sample_id: ID of the sample
            data: Data to save
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _check_existing_predictions(self, molecule_id: str) -> Optional[Dict]:
        """Check if MMST predictions already exist for a given molecule ID.
        
        Args:
            molecule_id: ID of the molecule to check
            
        Returns:
            Dict containing prediction results if they exist, None otherwise
        """
        try:
            # Load intermediate data
            intermediate_data = self._load_or_create_intermediate(molecule_id)
            
            # Check if molecule exists and has MMST predictions
            if ('molecule_data' in intermediate_data and 
                'mmst_results' in intermediate_data['molecule_data']):
                return {
                    'status': 'success',
                    'message': 'Retrieved existing predictions',
                    'predictions': intermediate_data['molecule_data']['mmst_results']
                }
            return None
        except Exception:
            return None

    async def _process_mmst_results(self, final_results_file: Path) -> Dict:
        """Process MMST results from all runs and combine them."""
        try:
            with open(final_results_file, 'r') as f:
                final_results = json.load(f)
                self.logger.info(f"Number of runs in final_results: {len(final_results.get('runs', []))}")
                self.logger.info(f"Content of final_results: {json.dumps(final_results, indent=2)}")

            # Initialize combined results structure
            combined_results = {
                "mmst_results": {
                    "generated_analogues_target": {},
                    "generatedSmilesProbabilities": {},
                    "generated_molecules": [],
                    "performance": 0.0,
                    "model_info": {
                        "model_path": final_results['runs'][-1]['model_save_path'] if final_results.get('runs') else ""
                    },
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "runs": final_results.get('runs', [])
                }
            }

            total_performance = 0
            unique_smiles_set = set()  # Track unique SMILES for deduplication
            molecule_data_dict = {}  # Track complete molecule data

            # Process each run's experimental results
            for run in final_results.get('runs', []):
                exp_results_file = run.get('exp_results_file')
                if not exp_results_file or not os.path.exists(exp_results_file):
                    self.logger.warning(f"Experimental results file not found: {exp_results_file}")
                    continue

                try:
                    with open(exp_results_file, 'r') as f:
                        run_data = json.load(f)
                    self.logger.info(f"Keys in run_data: {list(run_data.keys())}")
                    if "results" in run_data:
                        self.logger.info(f"Number of targets in results: {len(run_data['results'])}")
                        
                    if "results" in run_data:
                        # Process each target and its generated analogues
                        for target_smiles, analogues in run_data["results"].items():
                            # Canonicalize target SMILES
                            target_mol = Chem.MolFromSmiles(target_smiles)
                            if target_mol is None:
                                continue
                            canon_target = Chem.MolToSmiles(target_mol, canonical=True)
                            
                            if canon_target not in combined_results["mmst_results"]["generated_analogues_target"]:
                                combined_results["mmst_results"]["generated_analogues_target"][canon_target] = []
                                combined_results["mmst_results"]["generatedSmilesProbabilities"][canon_target] = []
                            
                            # Process and canonicalize each analogue
                            for analogue_data in analogues[0]:  # First level of nesting
                                try:
                                    # analogue_data = analogue_group[0]  # Second level - get the actual data
                                    smiles = analogue_data[0]
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol is None:
                                        continue
                                    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                                    
                                    # Only process if it's a new unique SMILES
                                    if canon_smiles not in unique_smiles_set:
                                        unique_smiles_set.add(canon_smiles)
                                        
                                        # Add to generated_analogues_target (only SMILES)
                                        if canon_smiles not in combined_results["mmst_results"]["generated_analogues_target"][canon_target]:
                                            combined_results["mmst_results"]["generated_analogues_target"][canon_target].append(canon_smiles)
                                        
                                        # Add probabilities
                                        probabilities = analogue_data[3] if isinstance(analogue_data[3], list) else analogue_data[3].tolist()
                                        if probabilities not in combined_results["mmst_results"]["generatedSmilesProbabilities"][canon_target]:
                                            combined_results["mmst_results"]["generatedSmilesProbabilities"][canon_target].append(probabilities)
                                        
                                        # Store complete molecule data
                                        molecule_data_dict[canon_smiles] = {
                                            "smiles": canon_smiles,
                                            "cosine_sim": float(analogue_data[1] if not isinstance(analogue_data[1], list) else analogue_data[1][0]),
                                            "dot_sim": float(analogue_data[2] if not isinstance(analogue_data[2], list) else analogue_data[2][0]),
                                            "probabilities": probabilities,
                                            "tanimoto_sim": float(analogue_data[4] if not isinstance(analogue_data[4], list) else analogue_data[4][0]),
                                            "HSQC_COSY_error": [float(x) if not isinstance(x, list) else float(x[0]) for x in analogue_data[5]]
                                        }
                                            
                                except Exception as e:
                                    self.logger.warning(f"Error processing SMILES {smiles}: {str(e)}")

                    # Add performance to average calculation
                    if "performance" in run_data:
                        total_performance += run_data["performance"]

                except Exception as e:
                    self.logger.warning(f"Error processing run results from {exp_results_file}: {str(e)}")
                    continue

            # Calculate average performance across all successful runs
            num_runs = len(final_results.get('runs', []))
            if num_runs > 0:
                combined_results["mmst_results"]["performance"] = total_performance / num_runs

            # Add the complete molecule data to generated_molecules
            combined_results["mmst_results"]["generated_molecules"] = list(molecule_data_dict.values())

            self.logger.info(f"Successfully combined results from {num_runs} runs")
            self.logger.info(f"Total unique molecules generated: {len(unique_smiles_set)}")
            
            return combined_results
        except Exception as e:
            self.logger.error(f"Error processing MMST results: {str(e)}")
            raise

    async def _execute_mmst_local(self, input_file: Path, molecule_id: Optional[str] = None) -> None:
        """Execute MMST prediction locally."""
        try:
            # Get sample directories
            sample_dirs = self.get_sample_dirs(molecule_id)
            
            # Create command with run directory
            cmd = [
                str(LOCAL_SCRIPT),
                f"--run_dir={str(self.run_dir)}",
                f"--input_csv={str(input_file)}",
                "--config_dir=/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/utils_MMT",
                f"--model_save_dir={str(sample_dirs['models'])}",
                f"--sgnn_gen_folder={str(sample_dirs['sgnn_output'])}",
                "--exp_data_path=/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/data/molecular_data/molecular_data.json"
            ]
            
            # Create a log file for this run
            log_file = sample_dirs['sample'] / f"mmst_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            self.logger.info(f"Logging MMST execution to: {log_file}")
            
            try:
                # Open log file for writing
                with open(log_file, 'w') as f:
                    f.write(f"=== MMST Execution Log ===\n")
                    f.write(f"Start Time: {datetime.now().isoformat()}\n")
                    f.write(f"Input CSV: {input_file}\n")
                    f.write(f"Output Dir: {self.temp_dir}\n\n")
                    f.write(f"Run Directory: {self.run_dir}\n\n")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Create async tasks to read stdout and stderr
                async def log_output(stream, prefix, log_file):
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        line_str = line.decode().strip()
                        # Log to file
                        with open(log_file, 'a') as f:
                            f.write(f"{prefix}: {line_str}\n")
                        # Also log to console
                        if prefix == 'stdout':
                            self.logger.info(line_str)
                        else:
                            self.logger.warning(line_str)
                
                # Start logging tasks
                stdout_task = asyncio.create_task(log_output(process.stdout, 'stdout', log_file))
                stderr_task = asyncio.create_task(log_output(process.stderr, 'stderr', log_file))
                
                # Wait for process to complete and logging tasks to finish
                await process.wait()
                await stdout_task
                await stderr_task
                
                # Log completion status
                with open(log_file, 'a') as f:
                    f.write(f"\nEnd Time: {datetime.now().isoformat()}\n")
                    f.write(f"Return Code: {process.returncode}\n")
                
                if process.returncode != 0:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    error_msg = f"MMST execution failed. Check log file: {log_file}\n\nLast few lines:\n{log_content[-500:]}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = f"Failed to execute MMST script: {str(e)}"
                self.logger.error(error_msg)
                # Log the error
                with open(log_file, 'a') as f:
                    f.write(f"\nError occurred: {error_msg}\n")
                raise RuntimeError(error_msg)
            
            self.logger.info(f"MMST execution completed. Full log available at: {log_file}")
        
        except Exception as e:
            self.logger.error(f"Error executing MMST locally: {str(e)}")
            raise

    async def predict_structure(self, reference_smiles: str, molecule_id: str = None, context: Optional[Dict[str, Any]] = None) -> Dict:
        """Predict molecular structure using MMST.
        
        Args:
            reference_smiles: SMILES string of reference molecule
            molecule_id: Optional molecule identifier
            context: Additional context for prediction
            
        Returns:
            Dict containing status and results/error message
        """
        self.logger.info("Starting MMST structure prediction")
        
        try:
            # Check for existing predictions first
            if molecule_id:
                existing = self._check_existing_predictions(molecule_id)
                if existing:
                    self.logger.info(f"Found existing predictions for {molecule_id}")
                    return existing

            # Create run ID
            run_id = str(uuid.uuid4())
            self.logger.info(f"Starting MMST prediction with run ID: {run_id}")
            
            # Create sample directory structure
            sample_dirs = self.get_sample_dirs(molecule_id)
            
            # Prepare input file
            input_file = await self._prepare_input_file(reference_smiles, molecule_id)
            
            # Check if using SLURM
            use_slurm = False  # Default to local execution
            if context and context.get('use_slurm'):
                use_slurm = True
            
            # Check CUDA availability for local execution
            try:
                import torch
                if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                    self.logger.warning("CUDA not available. Switching to SLURM execution.")
                    use_slurm = True
            except ImportError:
                self.logger.warning("PyTorch not available. Switching to SLURM execution.")
                use_slurm = True
                
            if use_slurm:
                # Execute using SBATCH
                self.logger.info("Executing MMST prediction using SLURM")
                job_id = await execute_sbatch(
                    str(SBATCH_SCRIPT),
                    f"--input_csv={str(input_file)}",
                    f"--output_dir={str(self.temp_dir)}",
                    f"--model_save_dir={str(self.temp_dir / 'models')}"
                )
                
                # Wait for job completion
                await wait_for_job_completion(job_id)
            else:
                # Execute locally
                self.logger.info("Executing MMST prediction locally")
                await self._execute_mmst_local(input_file, molecule_id)
            
            # Wait for output file
            output_file = await self._wait_for_output(molecule_id)
            
            # Process MMST results
            results = await self._process_mmst_results(output_file)
            
            # Save results to intermediate file
            intermediate_data = self._load_or_create_intermediate(molecule_id, context)
            intermediate_data['molecule_data']['mmst_results'] = results['mmst_results']
            self._save_intermediate(molecule_id, intermediate_data)
            
            return {
                'status': 'success',
                'message': 'Successfully predicted structure',
                'predictions': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in MMST prediction: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }