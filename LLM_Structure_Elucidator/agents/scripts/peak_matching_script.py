import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List
import asyncio

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TOOLS_DIR = PROJECT_ROOT / "LLM_Structure_Elucidator" / "agents" / "tools"
TEMP_DIR = PROJECT_ROOT / "LLM_Structure_Elucidator" / "_temp_folder" / "peak_matching"
CURRENT_RUN_DIR = TEMP_DIR / "current_run"
print("PROJECT_ROOT")
print(PROJECT_ROOT)
# Constants for peak matching
SUPPORTED_MATCHING_MODES = ['hung_dist_nn', 'euc_dist_all']
SUPPORTED_ERROR_TYPES = ['sum', 'avg']

# Configure logging
log_file = CURRENT_RUN_DIR / 'peak_matching.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file}")

# Add project root and tools directory to path for direct import
project_root_path = str(PROJECT_ROOT)
tools_path = str(TOOLS_DIR)
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
    logger.info(f"Added to Python path: {project_root_path}")
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)
    logger.info(f"Added to Python path: {tools_path}")

# Import required utilities
from utils_MMT.agents_code_v15_4_3 import generate_shifts_batch, add_atom_index_column
from utils_MMT import similarity_functions_exp_v15_4 as sfe

def extract_values(data_dict: Dict, key: str) -> List:
    """Extract values from potentially nested dictionary structures."""
    if not isinstance(data_dict, dict):
        return []
    if not data_dict.get(key):
        return []
    if isinstance(data_dict[key], dict):
        return list(data_dict[key].values())
    return data_dict[key]

async def compare_peaks(
    data1: Union[str, Dict],
    data2: Union[str, Dict],
    data_type: str,
    spectrum_type: str,
    config: Any,
    matching_mode: str = 'hung_dist_nn',
    error_type: str = 'sum'
) -> Dict:
    """Compare peaks between two inputs."""
    try:
        logger.info(f"Comparing peaks for {data_type} with spectrum type {spectrum_type}")
        
        # Validate input parameters
        if matching_mode not in SUPPORTED_MATCHING_MODES:
            raise ValueError(f"Unsupported matching mode: {matching_mode}. Must be one of {SUPPORTED_MATCHING_MODES}")
        if error_type not in SUPPORTED_ERROR_TYPES:
            raise ValueError(f"Unsupported error type: {error_type}. Must be one of {SUPPORTED_ERROR_TYPES}")
        
        logger.info(f"Using matching mode: {matching_mode}, error type: {error_type}")
        
        # Process data based on type
        if data_type == 'smiles_vs_smiles':
            # Initialize config for NMR generation with all required SGNN parameters
            config_dict = {
                'log_file': str(log_file),
                'output_directory': str(TEMP_DIR),
                'spectrum_type': spectrum_type.upper(),
                # SGNN required parameters
                'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                'SGNN_size_filter': 550,  # Maximum molecular weight filter
                'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                'data_type': 'sgnn'
            }
            
            # Create necessary directories
            os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
            os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
            os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
            
            # Generate NMR data for both SMILES
            logger.info("Generating NMR data for SMILES comparison")
            nmr_data1, _, _, _ = generate_shifts_batch(config_dict, [data1])
            nmr_data2, _, _, _ = generate_shifts_batch(config_dict, [data2])
            peaks1 = nmr_data1[0][spectrum_type.upper()]
            peaks2 = nmr_data2[0][spectrum_type.upper()]
        
        elif data_type == 'smiles_vs_peaks':
            # Initialize config for NMR generation with all required SGNN parameters
            config_dict = {
                'log_file': str(log_file),
                'output_directory': str(TEMP_DIR),
                'spectrum_type': spectrum_type.upper(),
                # SGNN required parameters
                'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                'SGNN_size_filter': 550,  # Maximum molecular weight filter
                'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                'data_type': 'sgnn'
            }
            
            # Create necessary directories
            os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
            os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
            os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
            
            # Generate NMR data for SMILES and use provided peaks
            logger.info("Generating NMR data for SMILES vs peaks comparison")
            nmr_data1, _, _, _ = generate_shifts_batch(config_dict, [data1])
            peaks1 = nmr_data1[0][spectrum_type.upper()]
            peaks2 = data2.get(spectrum_type.upper(), {})
        
        else:  # peaks_vs_peaks
            # Use provided peaks directly
            logger.info("Using provided peaks for comparison")
            peaks1 = data1.get(spectrum_type.upper(), {})
            peaks2 = data2.get(spectrum_type.upper(), {})

        logger.info("Converting data to DataFrames")
        
        # Handle potentially nested HSQC data
        if spectrum_type.upper() == 'HSQC':
            logger.info("Processing HSQC data")
            # Handle peaks1
            if isinstance(peaks1, dict):
                peaks1 = peaks1.get('HSQC', peaks1)
            if not peaks1:
                raise ValueError("No HSQC data found in peaks1")
                
            # Handle peaks2
            if isinstance(peaks2, dict):
                peaks2 = peaks2.get('HSQC', peaks2)
            if not peaks2:
                raise ValueError("No HSQC data found in peaks2")

        try:
            if spectrum_type.upper() in ['1H', '13C']:
                # Extract shifts for both peaks
                shifts1 = extract_values(peaks1, 'shifts')
                shifts2 = extract_values(peaks2, 'shifts')
                
                if not shifts1 or not shifts2:
                    logger.error(f"No shifts found in peaks for {spectrum_type}")
                    return {
                        'status': 'error',
                        'error': f'No shifts found for {spectrum_type}',
                        'type': 'data_error'
                    }
                
                # Convert to numpy arrays
                f1_ppm1 = np.array(shifts1, dtype=float)
                f1_ppm2 = np.array(shifts2, dtype=float)
                
                # Handle intensities based on spectrum type
                if spectrum_type.upper() == '1H':
                    # For 1H NMR, get intensities from data or use default 1.0
                    intensity1_raw = extract_values(peaks1, 'Intensity') or [1.0] * len(f1_ppm1)
                    intensity2_raw = extract_values(peaks2, 'Intensity') or [1.0] * len(f1_ppm2)
                else:  # 13C
                    # For 13C NMR, always use 1.0 for intensities
                    intensity1_raw = [1.0] * len(f1_ppm1)
                    intensity2_raw = [1.0] * len(f1_ppm2)
                
                # Convert intensities to numpy arrays
                intensity1_raw = np.array(intensity1_raw, dtype=float)
                intensity2_raw = np.array(intensity2_raw, dtype=float)
                
                # Store normalized intensity (all 1.0) for matching
                intensity1 = np.ones_like(intensity1_raw, dtype=float)
                intensity2 = np.ones_like(intensity2_raw, dtype=float)
                
                # Handle atom indices
                atom_idx1 = extract_values(peaks1, 'atom_index') or list(range(len(f1_ppm1)))
                atom_idx2 = extract_values(peaks2, 'atom_index') or list(range(len(f1_ppm2)))
                atom_idx1 = np.array(atom_idx1, dtype=int)
                atom_idx2 = np.array(atom_idx2, dtype=int)
                
                # Create DataFrames
                df1 = pd.DataFrame({
                    'shifts': f1_ppm1,
                    'Intensity': intensity1,  # Normalized intensity for matching
                    'actual_intensity': intensity1_raw,  # Actual intensity for reference
                    'atom_index': atom_idx1
                })
                
                df2 = pd.DataFrame({
                    'shifts': f1_ppm2,
                    'Intensity': intensity2,  # Normalized intensity for matching
                    'actual_intensity': intensity2_raw,  # Actual intensity for reference
                    'atom_index': atom_idx2
                })
            else:
                # For 2D spectra (HSQC, COSY)
                logger.info(f"Processing 2D spectrum type: {spectrum_type}")
                
                # Extract and validate F1 dimension
                f1_ppm1 = np.array(extract_values(peaks1, 'F1 (ppm)'), dtype=float)
                f1_ppm2 = np.array(extract_values(peaks2, 'F1 (ppm)'), dtype=float)
                if len(f1_ppm1) == 0 or len(f1_ppm2) == 0:
                    raise ValueError(f"Missing F1 dimension data for {spectrum_type}")
                
                # Extract and validate F2 dimension
                f2_ppm1 = np.array(extract_values(peaks1, 'F2 (ppm)'), dtype=float)
                f2_ppm2 = np.array(extract_values(peaks2, 'F2 (ppm)'), dtype=float)
                if len(f2_ppm1) == 0 or len(f2_ppm2) == 0:
                    raise ValueError(f"Missing F2 dimension data for {spectrum_type}")
                
                # Handle intensities with proper validation
                raw_intensity1 = extract_values(peaks1, 'Intensity')
                raw_intensity2 = extract_values(peaks2, 'Intensity')
                
                # Normalize intensities if present, otherwise use uniform weights
                if raw_intensity1:
                    intensity1 = np.array(raw_intensity1, dtype=float)
                    intensity1 = intensity1 / np.max(intensity1)  # Normalize to [0,1]
                else:
                    intensity1 = np.ones(len(f1_ppm1), dtype=float)
                
                if raw_intensity2:
                    intensity2 = np.array(raw_intensity2, dtype=float)
                    intensity2 = intensity2 / np.max(intensity2)  # Normalize to [0,1]
                else:
                    intensity2 = np.ones(len(f1_ppm2), dtype=float)
                
                # Handle atom indices
                atom_idx1 = np.array(extract_values(peaks1, 'atom_index') or list(range(len(f1_ppm1))), dtype=int)
                atom_idx2 = np.array(extract_values(peaks2, 'atom_index') or list(range(len(f1_ppm2))), dtype=int)
                
                # Create DataFrames with validated data
                df1 = pd.DataFrame({
                    'F1 (ppm)': f1_ppm1,
                    'F2 (ppm)': f2_ppm1,
                    'Intensity': intensity1,
                    'atom_index': atom_idx1
                })
                
                df2 = pd.DataFrame({
                    'F1 (ppm)': f1_ppm2,
                    'F2 (ppm)': f2_ppm2,
                    'Intensity': intensity2,
                    'atom_index': atom_idx2
                })
                
                logger.info(f"Created DataFrames for 2D spectra comparison: df1 shape {df1.shape}, df2 shape {df2.shape}")

        except (TypeError, ValueError) as e:
            logger.error(f"Error converting peak data: {str(e)}")
            logger.error(f"Peaks1 data: {peaks1}")
            logger.error(f"Peaks2 data: {peaks2}")
            raise

        logger.info("Calculating_similarity")
        # Calculate similarity using the unified calculation
        overall_error, df1_processed, df2_processed = sfe.unified_similarity_calculation(
            df1, df2, 
            spectrum_type.upper(),
            method=matching_mode,
            error_type=error_type
        )
        return {
            'status': 'success',
            'overall_error': float(overall_error),
            'spectrum_type': spectrum_type,
            'data_type': data_type,
            'error_type': error_type,
            'matching_mode': matching_mode,
            'matched_peaks': {
                'spectrum1': df1_processed.to_dict('records'),
                'spectrum2': df2_processed.to_dict('records')
            },
            'original_data': {
                'spectrum1': df1.to_dict('records'),
                'spectrum2': df2.to_dict('records')
            }
        }
            
    except Exception as e:
        logger.error(f"Error in peak comparison: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'status': 'error',
            'error': str(e),
            'type': 'comparison_error'
        }

async def process_peak_matching(input_path: str) -> Dict[str, Any]:
    """Process peak matching based on input data."""
    logger.info(f"Starting process_peak_matching with input: {input_path}")
    
    try:
        logger.info("Reading input data")
        with open(input_path) as f:
            input_data = json.load(f)
        logger.info(f"Input data loaded: {json.dumps(input_data, indent=2)}")
        
        # Extract parameters
        logger.info("Extracting parameters")
        data_type = input_data['type']
        spectra = input_data['spectra']
        matching_mode = input_data.get('matching_mode', 'hung_dist_nn')
        error_type = input_data.get('error_type', 'sum')
        
        # Process based on data type
        logger.info(f"Processing {data_type} comparison")
        
        results = {}
        for spectrum_type in spectra:
            if data_type == 'smiles_vs_smiles':
                # Initialize config for NMR generation with all required SGNN parameters
                config_dict = {
                    'log_file': str(log_file),
                    'output_directory': str(TEMP_DIR),
                    'spectrum_type': spectrum_type.upper(),
                    # SGNN required parameters
                    'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                    'SGNN_size_filter': 550,  # Maximum molecular weight filter
                    'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                    'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                    'data_type': 'sgnn'
                }
                
                # Create necessary directories
                os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
                os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
                os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
                
                result = await compare_peaks(
                    input_data['smiles1'],
                    input_data['smiles2'],
                    data_type,
                    spectrum_type,
                    config_dict,
                    matching_mode,
                    error_type
                )
            elif data_type == 'smiles_vs_peaks':
                # Initialize config for NMR generation with all required SGNN parameters
                config_dict = {
                    'log_file': str(log_file),
                    'output_directory': str(TEMP_DIR),
                    'spectrum_type': spectrum_type.upper(),
                    # SGNN required parameters
                    'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                    'SGNN_size_filter': 550,  # Maximum molecular weight filter
                    'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                    'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                    'data_type': 'sgnn'
                }
                
                # Create necessary directories
                os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
                os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
                os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
                
                result = await compare_peaks(
                    input_data['smiles'],
                    input_data['peaks'],
                    data_type,
                    spectrum_type,
                    config_dict,
                    matching_mode,
                    error_type
                )
            elif data_type == 'peaks_vs_peaks':
                result = await compare_peaks(
                    input_data['peaks1'],
                    input_data['peaks2'],
                    data_type,
                    spectrum_type,
                    None,  # TODO: Add config if needed
                    matching_mode,
                    error_type
                )
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
            results[spectrum_type] = result
        
        # Prepare final result
        final_result = {
            'status': 'success',
            'data': {
                'type': data_type,
                'spectra': spectra,
                'matching_mode': matching_mode,
                'error_type': error_type,
                'results': results
            }
        }
        
        # Save results
        logger.info("Saving results")
        result_path = Path(input_path).parent / 'results.json'
        with open(result_path, 'w') as f:
            json.dump(final_result, f, indent=2)
            
        logger.info(f"Results saved to {result_path}")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in peak matching process: {str(e)}", exc_info=True)
        error_result = {
            'status': 'error',
            'error': str(e),
            'type': 'peak_matching_error'
        }
        
        # Save error result
        result_path = Path(input_path).parent / 'results.json'
        with open(result_path, 'w') as f:
            json.dump(error_result, f, indent=2)
            
        return error_result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python peak_matching_script.py <input_json_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
        
    # Run the async function
    result = asyncio.run(process_peak_matching(input_path))
