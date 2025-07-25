"""Tool for peak matching between NMR spectra with support for various input formats."""
import os
import logging
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys
import pandas as pd
import asyncio

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
TEMP_DIR = BASE_DIR / "_temp_folder"
PEAK_MATCHING_DIR = TEMP_DIR / "peak_matching"
LOCAL_SCRIPT = SCRIPTS_DIR / "peak_matching_local.sh"

# Constants for peak matching
SUPPORTED_SPECTRA = ['1H', '13C', 'HSQC', 'COSY']
OUTPUT_CHECK_INTERVAL = 5  # seconds
OUTPUT_TIMEOUT = 300  # 5 minutes
RESULTS_FILENAME = "results.json"

# Constants for timeouts and retries
SUBPROCESS_TIMEOUT = 300  # 5 minutes
RESULTS_WAIT_TIMEOUT = 60  # 1 minute
SUBPROCESS_CHECK_INTERVAL = 0.5  # 500ms
RESULTS_CHECK_INTERVAL = 1.0  # 1 second
MAX_RETRIES = 3

# Constants for peak matching configuration
SUPPORTED_MATCHING_MODES = ['hung_dist_nn', 'euc_dist_all']  # Supported peak matching strategies
SUPPORTED_ERROR_TYPES = ['sum', 'avg']  # Supported error calculation methods
DEFAULT_MATCHING_MODE = 'hung_dist_nn'
DEFAULT_ERROR_TYPE = 'avg'

"""
Future enhancements for peak matching configuration:
- Add intensity weighting for peak matching
- Add distance normalization options
- Add maximum distance thresholds
- Add weighted averaging for error calculation
- Add configurable parameters for each matching mode
"""

class EnhancedPeakMatchingTool:
    """Tool for comparing NMR peak lists using external Python environment."""
    
    def __init__(self):
        """Initialize the peak matching tool with required directories."""
        self.scripts_dir = SCRIPTS_DIR
        self.temp_dir = TEMP_DIR
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        self.peak_matching_dir = PEAK_MATCHING_DIR
        
        # Create directories if they don't exist
        self.temp_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        self.peak_matching_dir.mkdir(exist_ok=True)
        
        # Validate script existence
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required local script not found at {LOCAL_SCRIPT}")
        
        # Configure logging
        log_file = PEAK_MATCHING_DIR / 'enhanced_tool.log'
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EnhancedPeakMatchingTool initialized. Log file: {log_file}")


    def _validate_simulation_data(self, data: Dict) -> bool:
        """Check if required simulation data exists in NMR data"""
        if 'molecule_data' not in data:
            return False
        
        mol_data = data['molecule_data']
        if 'nmr_data' not in mol_data:
            return False
        
        required_keys = ['1H_sim', '13C_sim', 'HSQC_sim', 'COSY_sim']
        nmr_data = mol_data['nmr_data']
        
        for key in required_keys:
            if key not in nmr_data:
                self.logger.warning(f"Missing required simulation data: {key}")
                return False
            if not nmr_data[key]:  # Check if data exists
                self.logger.warning(f"Empty simulation data for: {key}")
                return False
                
        return True
        
        
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
        """Save data to intermediate file"""
        intermediate_path = self.intermediate_dir / f"{sample_id}_intermediate.json"
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _prepare_peak_matching_input(self, mol_data: Dict) -> Dict:
        """Prepare input data for peak matching from molecule data"""
        if 'nmr_data' not in mol_data:
            raise ValueError("No NMR data found in molecule data")
        
        nmr_data = mol_data['nmr_data']
        
        # Format peaks for 1D NMR (1H)
        def format_1d_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # Each peak is a list of [shift, intensity]
            return {
                'shifts': [peak[0] for peak in peaks],
                #'intensities': [peak[1] for peak in peaks]
                'intensities': [1 for peak in peaks] # Constant intensity
            }
        
        # Format peaks for 2D NMR (HSQC, COSY)
        def format_2d_peaks(peaks):
            if not peaks:
                return {'F2 (ppm)': [], 'F1 (ppm)': []}
            # Each peak is a list of [f2, f1]
            return {
                'F2 (ppm)': [peak[0] for peak in peaks],
                'F1 (ppm)': [peak[1] for peak in peaks]
            }
        
        # Format peaks for 13C NMR
        def format_13c_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # 13C peaks are just a list of shifts
            return {
                'shifts': peaks,
                'intensities': [1.0] * len(peaks)  # Constant intensity for 13C
            }
    
        # Prepare input data structure
        input_data = {
            'spectra': SUPPORTED_SPECTRA,
            'simulated_data': {
                'h1': format_1d_peaks(nmr_data.get('1H_sim', [])),
                'c13': format_13c_peaks(nmr_data.get('13C_sim', [])),
                'hsqc': format_2d_peaks(nmr_data.get('HSQC_sim', [])),
                'cosy': format_2d_peaks(nmr_data.get('COSY_sim', []))
            },
            'experimental_data': {
                'h1': format_1d_peaks(nmr_data.get('1H_exp', [])),
                'c13': format_13c_peaks(nmr_data.get('13C_exp', [])),
                'hsqc': format_2d_peaks(nmr_data.get('HSQC_exp', [])),
                'cosy': format_2d_peaks(nmr_data.get('COSY_exp', []))
            }
        }
    
        return input_data

    async def _wait_for_results(self, run_dir: Path, context: Dict) -> Dict:
        """Wait for and validate results file.
        
        Args:
            run_dir: Directory containing results
            context: Additional context dictionary
            
        Returns:
            Dictionary containing results
            
        Raises:
            TimeoutError: If results not found within timeout period
            ValueError: If results file is invalid
        """
        start_time = datetime.now()
        results_path = run_dir / RESULTS_FILENAME
        
        while not results_path.exists():
            if (datetime.now() - start_time).total_seconds() > RESULTS_WAIT_TIMEOUT:
                error_msg = f"Timeout waiting for results file after {RESULTS_WAIT_TIMEOUT} seconds"
                self.logger.error(error_msg)
                # Check for error file
                error_file = run_dir / 'error.log'
                if error_file.exists():
                    with open(error_file) as f:
                        error_content = f.read()
                    error_msg += f"\nError log content:\n{error_content}"
                raise TimeoutError(error_msg)
            await asyncio.sleep(RESULTS_CHECK_INTERVAL)
        
        # Validate results file
        with open(results_path) as f:
            results = json.load(f)
            return results
            
    def _has_existing_results(self, intermediate_data: Dict, comparison_type: str = 'peaks_vs_peaks') -> bool:
        """Check if peak matching results already exist for the given comparison type."""
        if 'molecule_data' not in intermediate_data or \
           'peak_matching_results' not in intermediate_data['molecule_data'] or \
           'comparisons' not in intermediate_data['molecule_data']['peak_matching_results']:
            return False
            
        # Determine category and subcategory
        if comparison_type == 'peaks_vs_peaks':
            category = 'simulation'
            subcategory = 'exp_vs_sim'
        elif comparison_type == 'smiles_vs_peaks':
            # This will be determined by context in _save_results
            return False  # Always run for SMILES comparisons
        else:
            category = 'custom'
            subcategory = comparison_type
            
        comparisons = intermediate_data['molecule_data']['peak_matching_results']['comparisons']
        return (category in comparisons and 
                subcategory in comparisons[category] and
                comparisons[category][subcategory]['status'] == 'success')

    def _get_existing_results(self, intermediate_data: Dict, comparison_type: str = 'peaks_vs_peaks') -> Dict:
        """Get existing peak matching results for the given comparison type."""
        if comparison_type == 'peaks_vs_peaks':
            category = 'simulation'
            subcategory = 'exp_vs_sim'
        else:
            category = 'custom'
            subcategory = comparison_type
            
        results = intermediate_data['molecule_data']['peak_matching_results']['comparisons'][category][subcategory]
        return {
            'status': 'success',
            'message': f'Peak matching results already exist for {category}/{subcategory}',
            'data': {
                'type': comparison_type,
                'results': results['results'],
                # 'matching_mode': results['metadata']['matching_mode'],
                # 'error_type': results['metadata']['error_type'],
                # 'spectra': results['metadata']['spectra']
            }
        }

    def _save_results(self, intermediate_data: Dict, results: Dict) -> None:
        """Save peak matching results to intermediate file with structured format."""
        # Initialize peak_matching_results if it doesn't exist
        if 'peak_matching_results' not in intermediate_data['molecule_data']:
            intermediate_data['molecule_data']['peak_matching_results'] = {
                'comparisons': {}
            }
        
        # Get the comparison type from results
        comparison_type = results.get('data', {}).get('type', 'unknown')
        
        # Determine the category and subcategory based on comparison type
        if comparison_type == 'peaks_vs_peaks':
            category = 'simulation'
            subcategory = 'exp_vs_sim'
        elif comparison_type == 'smiles_vs_peaks':
            # Check context to determine if it's mol2mol or mmst
            if 'mol2mol' in str(results.get('source', '')).lower():
                category = 'structure_candidates'
                subcategory = 'mol2mol'
            elif 'mmst' in str(results.get('source', '')).lower():
                category = 'structure_candidates'
                subcategory = 'mmst'
            else:
                category = 'custom'
                subcategory = 'smiles_peaks'
        else:
            category = 'custom'
            subcategory = comparison_type
        
        # Initialize category if it doesn't exist
        if category not in intermediate_data['molecule_data']['peak_matching_results']['comparisons']:
            intermediate_data['molecule_data']['peak_matching_results']['comparisons'][category] = {}
        
        # Save results in the appropriate category
        intermediate_data['molecule_data']['peak_matching_results']['comparisons'][category][subcategory] = {
            'status': results['status'],
            'timestamp': datetime.now().isoformat(),
            'results': results.get('data', {}).get('results', {}),
            # 'metadata': {
            #     'matching_mode': results.get('data', {}).get('matching_mode', ''),
            #     'error_type': results.get('data', {}).get('error_type', ''),
            #     'spectra': results.get('data', {}).get('spectra', [])
            # }
        }
        
        # # Update last_updated timestamp
        # intermediate_data['molecule_data']['peak_matching_results']['metadata']['last_updated'] = \
        #     datetime.now().isoformat()
        
        # Save to file
        self._save_intermediate(intermediate_data['molecule_data']['sample_id'], intermediate_data)

    def _prepare_exp_vs_sim_input(self, intermediate_data: Dict) -> Dict:
        """Prepare input data for experimental vs simulated comparison."""
        if 'molecule_data' not in intermediate_data or 'nmr_data' not in intermediate_data['molecule_data']:
            raise ValueError("No NMR data found in intermediate data")
            
        nmr_data = intermediate_data['molecule_data']['nmr_data']
         
        # Format peaks for 1D NMR (1H)
        def format_1d_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # Each peak is a list of [shift, intensity]
            return {
                'shifts': [peak[0] for peak in peaks],
                'intensities': [1.0 for peak in peaks]  # Constant intensity
            }
        
        # Format peaks for 2D NMR (HSQC, COSY)
        def format_2d_peaks(peaks):
            if not peaks:
                return {'F2 (ppm)': [], 'F1 (ppm)': []}
            # Each peak is a list of [f2, f1]
            return {
                'F2 (ppm)': [peak[0] for peak in peaks],
                'F1 (ppm)': [peak[1] for peak in peaks]
            }
        
        # Format peaks for 13C NMR
        def format_13c_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # 13C peaks are just a list of shifts
            return {
                'shifts': peaks,
                'intensities': [1.0] * len(peaks)  # Constant intensity for 13C
            }
    
        # Format experimental and simulated data
        peaks1 = {
            '1H': format_1d_peaks(nmr_data.get('1H_exp', [])),
            '13C': format_13c_peaks(nmr_data.get('13C_exp', [])),
            'HSQC': format_2d_peaks(nmr_data.get('HSQC_exp', [])),
            'COSY': format_2d_peaks(nmr_data.get('COSY_exp', []))
        }
        
        peaks2 = {
            '1H': format_1d_peaks(nmr_data.get('1H_sim', [])),
            '13C': format_13c_peaks(nmr_data.get('13C_sim', [])),
            'HSQC': format_2d_peaks(nmr_data.get('HSQC_sim', [])),
            'COSY': format_2d_peaks(nmr_data.get('COSY_sim', []))
        }
    
        return {
            'type': 'peaks_vs_peaks',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'peaks1': peaks1,  # Experimental peaks
            'peaks2': peaks2   # Simulated peaks
        }

    def _prepare_smiles_comparison_input(self, context: Dict) -> Dict:
        """Prepare input data for SMILES vs SMILES comparison."""
        input_data = context.get('input_data', {})
        if 'smiles1' not in input_data or 'smiles2' not in input_data:
            raise ValueError("Missing SMILES data for comparison")
            
        return {
            'type': 'smiles_vs_smiles',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'smiles1': input_data['smiles1'],
            'smiles2': input_data['smiles2']
        }

    def _prepare_smiles_peaks_input(self, context: Dict) -> Dict:
        """Prepare input data for SMILES vs peaks comparison."""
        input_data = context.get('input_data', {})
        if 'smiles' not in input_data or 'peaks' not in input_data:
            raise ValueError("Missing SMILES or peaks data for comparison")
            
        return {
            'type': 'smiles_vs_peaks',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'smiles': input_data['smiles'],
            'peaks': input_data['peaks']
        }

    def _prepare_peaks_csv_input(self, context: Dict) -> Dict:
        """Prepare input data for peaks vs SMILES CSV comparison."""
        input_data = context.get('input_data', {})
        if 'peaks' not in input_data or 'smiles_csv' not in input_data:
            raise ValueError("Missing peaks or SMILES CSV data for comparison")
            
        return {
            'type': 'peaks_vs_smiles_csv',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'peaks': input_data['peaks'],
            'smiles_csv': input_data['smiles_csv']
        }

    async def _run_peak_matching(self, input_data: Dict) -> Dict:
        """Run peak matching script with prepared input data."""
        try:
            # Create run directory
            run_dir = self.peak_matching_dir / 'current_run'
            if run_dir.exists():
                self.logger.info(f"Cleaning up previous run directory {run_dir}")
                for file in run_dir.glob('*'):
                    file.unlink()
            run_dir.mkdir(exist_ok=True)
            
            # Save input data
            data_path = run_dir / 'input_data.json'
            with open(data_path, 'w') as f:
                json.dump(input_data, f, indent=2)
            
            # Execute peak matching script
            self.logger.info("Executing peak matching script")
            cmd = [str(LOCAL_SCRIPT), str(data_path)]
            
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(BASE_DIR)
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(),
                            timeout=SUBPROCESS_TIMEOUT
                        )
                        
                        if process.returncode != 0:
                            error_msg = f"Peak matching script failed with code {process.returncode}"
                            if stderr:
                                error_msg += f"\nError output:\n{stderr.decode()}"
                            raise RuntimeError(error_msg)
                        
                        # Get results
                        return await self._wait_for_results(run_dir, {})
                        
                    except asyncio.TimeoutError:
                        process.kill()
                        raise TimeoutError(f"Peak matching script timed out after {SUBPROCESS_TIMEOUT} seconds")
                        
                except Exception as e:
                    retries += 1
                    if retries >= MAX_RETRIES:
                        self.logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                        return {'status': 'error', 'message': str(e)}
                    self.logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Error running peak matching: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def process_peaks(self, sample_id: str, context: Optional[Dict] = None) -> Dict:
        """Process peak matching with different modes"""
        try:
            # Load intermediate data
            intermediate_data = self._load_or_create_intermediate(sample_id, context)
            
            # Check if results exist
            if self._has_existing_results(intermediate_data):
                return self._get_existing_results(intermediate_data)
                
            # Determine comparison mode
            comparison_mode = self._determine_comparison_mode(context)
            
            # Prepare input data based on mode
            if comparison_mode == 'default':
                input_data = self._prepare_exp_vs_sim_input(intermediate_data)
            elif comparison_mode == 'smiles_vs_smiles':
                input_data = self._prepare_smiles_comparison_input(context)
            elif comparison_mode == 'smiles_vs_peaks':
                input_data = self._prepare_smiles_peaks_input(context)
            elif comparison_mode == 'peaks_vs_smiles_csv':
                input_data = self._prepare_peaks_csv_input(context)
            else:
                raise ValueError(f"Unsupported comparison mode: {comparison_mode}")
                
            # Run peak matching
            results = await self._run_peak_matching(input_data)
            
            # Save results if successful
            if results['status'] == 'success':
                self._save_results(intermediate_data, results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in peak matching: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _determine_comparison_mode(self, context: Optional[Dict]) -> str:
        """Determine the comparison mode based on context."""
        if not context:
            return 'default'
            
        input_data = context.get('input_data', {})
        
        if 'comparison_mode' in input_data:
            return input_data['comparison_mode']
            
        if 'smiles1' in input_data and 'smiles2' in input_data:
            return 'smiles_vs_smiles'
        elif 'smiles' in input_data and 'peaks' in input_data:
            return 'smiles_vs_peaks'
        elif 'peaks' in input_data and 'smiles_csv' in input_data:
            return 'peaks_vs_smiles_csv'
            
        return 'default'

#---------------------------------------------------------------------------------

    async def _prepare_input_data(
        self,
        input_data: Dict,
        run_dir: Path,
        context: Dict
    ) -> Dict:
        """Prepare input data for peak matching script.
        
        Args:
            input_data: Input data dictionary containing:
                - Required: One of the following combinations:
                    * smiles1, smiles2 for SMILES vs SMILES comparison
                    * smiles, peaks for SMILES vs peaks comparison
                    * peaks1, peaks2 for peaks vs peaks comparison
                    * peaks, smiles_csv for peaks vs SMILES CSV comparison
                    * reference_smiles, smiles_csv for SMILES vs SMILES CSV comparison
                - Optional:
                    * matching_mode: Peak matching strategy ('hung_dist_nn' or 'euc_dist_all')
                    * error_type: Error calculation method ('sum' or 'avg')
                    * spectra: List of spectrum types to compare
            run_dir: Directory for this run
            context: Additional context dictionary
            
        Returns:
            Dictionary with prepared data and paths
        """
        try:
            self.logger.info("Preparing input data")
            
            # Validate peak matching configuration
            matching_mode = input_data.get('matching_mode', context.get('matching_mode', DEFAULT_MATCHING_MODE))
            if matching_mode not in SUPPORTED_MATCHING_MODES:
                raise ValueError(f"Unsupported matching mode: {matching_mode}. Must be one of {SUPPORTED_MATCHING_MODES}")
                
            error_type = input_data.get('error_type', context.get('error_type', DEFAULT_ERROR_TYPE))
            if error_type not in SUPPORTED_ERROR_TYPES:
                raise ValueError(f"Unsupported error type: {error_type}. Must be one of {SUPPORTED_ERROR_TYPES}")
            
            # Determine input type and prepare data
            if 'smiles1' in input_data and 'smiles2' in input_data:
                data_type = 'smiles_vs_smiles'
                prepared_data = {
                    'type': data_type,
                    'smiles1': input_data['smiles1'],
                    'smiles2': input_data['smiles2']
                }
            elif 'smiles' in input_data and 'peaks' in input_data:
                data_type = 'smiles_vs_peaks'
                prepared_data = {
                    'type': data_type,
                    'smiles': input_data['smiles'],
                    'peaks': input_data['peaks']
                }
            elif 'peaks1' in input_data and 'peaks2' in input_data:
                data_type = 'peaks_vs_peaks'
                prepared_data = {
                    'type': data_type,
                    'peaks1': input_data['peaks1'],
                    'peaks2': input_data['peaks2']
                }
            elif 'peaks' in input_data and 'smiles_csv' in input_data:
                data_type = 'peaks_vs_smiles_csv'
                prepared_data = {
                    'type': data_type,
                    'peaks': input_data['peaks'],
                    'smiles_csv': input_data['smiles_csv']
                }
            elif 'reference_smiles' in input_data and 'smiles_csv' in input_data:
                data_type = 'smiles_vs_smiles_csv'
                prepared_data = {
                    'type': data_type,
                    'reference_smiles': input_data['reference_smiles'],
                    'smiles_csv': input_data['smiles_csv']
                }
            else:
                raise ValueError("Invalid input data format")

            # Add context information
            prepared_data.update({
                'spectra': context.get('spectra', SUPPORTED_SPECTRA),
                'matching_mode': context.get('matching_mode', 'hung_dist_nn'),
                'error_type': context.get('error_type', 'sum')
            })
            
            # Note: The following configuration options are planned for future implementation:
            # config = {
            #     'matching': {
            #         'mode': matching_mode,
            #         'parameters': {
            #             'hung_dist_nn': {
            #                 'max_distance': 0.1,  # Maximum distance for peak matching
            #                 'use_intensity': True  # Whether to consider peak intensities
            #             },
            #             'euc_dist_all': {
            #                 'threshold': 0.1,  # Distance threshold for considering peaks as matching
            #                 'normalize': True  # Whether to normalize distances
            #             }
            #         }
            #     },
            #     'error': {
            #         'type': error_type,
            #         'parameters': {
            #             'sum': {},  # No additional parameters needed
            #             'avg': {
            #                 'weighted': True  # Whether to weight by peak intensities
            #             }
            #         }
            #     }
            # }
            
            # Save prepared data
            data_path = run_dir / 'input_data.json'
            with open(data_path, 'w') as f:
                json.dump(prepared_data, f, indent=2)
            return {
                'status': 'success',
                'data_type': data_type,
                'data_path': str(data_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing input data: {str(e)}")
            raise


    async def process(
        self,
        input_data: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process peak matching request."""
        self.logger.info("Starting peak matching tool process")
        context = context or {}
        
        # Create run directory
        run_dir = self.peak_matching_dir / 'current_run'
        if run_dir.exists():
            self.logger.info("Cleaning up previous run directory")
            for file in run_dir.glob('*'):
                file.unlink()
        run_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created run directory: {run_dir}")
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                # Prepare input data
                self.logger.info("Preparing input data")
                prep_result = await self._prepare_input_data(input_data, run_dir, context)
                if prep_result['status'] != 'success':
                    self.logger.error(f"Input data preparation failed: {prep_result}")
                    return prep_result
                self.logger.info("Input data preparation successful")
                
                # Execute peak matching script
                self.logger.info("Executing peak matching script")
                cmd = [str(LOCAL_SCRIPT), str(prep_result['data_path'])]
                self.logger.info(f"Command: {' '.join(cmd)}")
                
                try:
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(BASE_DIR)
                    self.logger.info(f"Environment: PYTHONPATH={env['PYTHONPATH']}")
                    
                    # Start subprocess with timeout
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(),
                            timeout=SUBPROCESS_TIMEOUT
                        )
                        
                        if process.returncode != 0:
                            error_msg = f"Peak matching script failed with code {process.returncode}"
                            if stderr:
                                error_msg += f"\nError output:\n{stderr.decode()}"
                            raise RuntimeError(error_msg)
                        
                        # Wait for and return results
                        return await self._wait_for_results(run_dir, {})
                        
                    except asyncio.TimeoutError:
                        process.kill()
                        raise TimeoutError(f"Peak matching script timed out after {SUBPROCESS_TIMEOUT} seconds")
                        
                except (subprocess.SubprocessError, OSError) as e:
                    error_msg = f"Error executing peak matching script: {str(e)}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    self.logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                    raise
                self.logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying...")
                await asyncio.sleep(1)  # Wait before retry