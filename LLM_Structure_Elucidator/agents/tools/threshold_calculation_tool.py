"""Tool for calculating thresholds using retrosynthesis, NMR simulation, and peak matching."""
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from .nmr_simulation_tool import NMRSimulationTool
from .retro_synthesis_tool import RetrosynthesisTool
from .peak_matching_tool import EnhancedPeakMatchingTool

class ThresholdCalculationTool:
    """Tool for calculating thresholds using multiple specialized tools."""
    
    def __init__(self):
        """Initialize the ThresholdCalculationTool."""
        self.retro_tool = RetrosynthesisTool()
        self.nmr_tool = NMRSimulationTool()
        self.peak_matching_tool = EnhancedPeakMatchingTool()
        self.master_data_path = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
        self.temp_dir = Path(__file__).parent.parent.parent / "_temp_folder"
        self.peak_matching_dir = self.temp_dir / "peak_matching"  # Updated to use correct path
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        self.peak_matching_dir.mkdir(exist_ok=True)  # Ensure directory exists
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set logger level to DEBUG
        
        
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
        master_path = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
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

    async def calculate_threshold(self, sample_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculate threshold using retrosynthesis and NMR simulation pipeline.
        
        Args:
            sample_id: ID of the sample to calculate thresholds for
            context: Optional context information for the calculation
            
        Returns:
            Dict containing the calculation results or error information
        """
        try:
            # Load or create intermediate data
            self.logger.info(f"Loading intermediate data for sample {sample_id}...")
            try:
                intermediate_data = self._load_or_create_intermediate(sample_id, context)
                self.logger.info("Successfully loaded intermediate data")
            except ValueError as e:
                self.logger.error(f"Failed to load intermediate data: {str(e)}")
                return {'status': 'error', 'message': str(e)}
            
            # Check if we already have threshold results
            if 'threshold_data' in intermediate_data:
                self.logger.info(f"Retrieved cached threshold results for sample {sample_id}")
                return {'status': 'success', 'data': intermediate_data['threshold_data']}

            # Get SMILES from intermediate data
            target_smiles = intermediate_data['molecule_data'].get('smiles')
            if not target_smiles:
                error_msg = f"No SMILES found in intermediate data for sample {sample_id}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}

            self.logger.info(f"Found SMILES for sample_id {sample_id}: {target_smiles}")
            
            try:
                # Step 1: Run retrosynthesis prediction
                self.logger.info(f"Starting retrosynthesis prediction for SMILES: {target_smiles}")
                
                try:
                    # Run retrosynthesis prediction
                    self.logger.info("Calling retrosynthesis prediction tool")
                    retro_result = await self.retro_tool.predict_retrosynthesis(
                        target_smiles,
                        context={'use_slurm': False}  # Force local execution
                    )
                    self.logger.info(f"Retrosynthesis prediction result status: {retro_result.get('status', 'unknown')}")
                    
                    # Reload molecular data to get updated predictions
                    self.logger.info("Reloading molecular data after retrosynthesis")
                    all_data = self._load_molecular_data()
                    
                    # Check if retrosynthesis results were stored for this molecule
                    if sample_id in all_data:
                        sample_data = all_data[sample_id]
                        starting_materials = sample_data.get('starting_smiles', [])
                        if starting_materials:  # Check if list is not empty
                            starting_material_smiles = starting_materials[0]
                            self.logger.info(f"Found starting material in master data: {starting_material_smiles}")
                        else:
                            self.logger.warning("No starting materials found in master data (empty list)")
                    else:
                        self.logger.warning(f"Sample {sample_id} not found in master data")
                        
                except Exception as e:
                    self.logger.error(f"Error in retrosynthesis prediction or data retrieval: {str(e)}")

                # If no starting material found in master data, use target molecule itself
                if not starting_material_smiles:
                    self.logger.info("Using target molecule as starting material for threshold calculation")
                    starting_material_smiles = target_smiles
                
                # Step 2: Run NMR simulation for target and starting materials
                self.logger.info(f"Running NMR simulation for target ({target_smiles}) and starting materials ({starting_material_smiles})")
                
                # Create a temporary JSON file for this specific simulation
                temp_json_path = self.temp_dir / f"temp_simulation_{sample_id}.json"
                
                # Split starting materials (they're separated by '.')
                starting_materials = starting_material_smiles.split('.')
                
                # Prepare simulation data structure - keep it simple with just required fields
                simulation_data = {}
                
                # Add target molecule
                simulation_data[f"{sample_id}_target"] = {
                    "smiles": target_smiles,
                    "sample_id": f"{sample_id}_target"
                }
                
                # Add starting materials
                for idx, start_mat in enumerate(starting_materials):
                    simulation_data[f"{sample_id}_starting_{idx}"] = {
                        "smiles": start_mat,
                        "sample_id": f"{sample_id}_starting_{idx}"
                    }
                
                # Write temporary JSON
                self.logger.debug(f"Writing simulation data to {temp_json_path}")
                self.logger.debug(f"Simulation data structure: {json.dumps(simulation_data, indent=2)}")
                with open(temp_json_path, 'w') as f:
                    json.dump(simulation_data, f, indent=2)
                
                try:
                    # Run simulation
                    self.logger.info("Starting NMR simulation")
                    sim_result = await self.nmr_tool.simulate(
                        str(temp_json_path),
                        context={'use_slurm': False}
                    )
                    self.logger.debug(f"NMR simulation result: {json.dumps(sim_result, indent=2)}")
                    
                    if sim_result['status'] != 'success':
                        error_msg = f"NMR simulation failed: {sim_result.get('message', 'Unknown error')}"
                        self.logger.error(error_msg)
                        return {
                            'status': 'error',
                            'message': error_msg
                        }
                    
                    # Load master data to get simulation results
                    self.logger.debug("Loading master data to retrieve simulation results")
                    master_data = self._load_molecular_data()

                    # Initialize threshold calculation data
                    self.logger.debug("Initializing threshold calculation data structure")
                    
                    # Get target simulation data from master data
                    target_key = f"{sample_id}_target"
                    self.logger.debug(f"Looking for target simulation data with key: {target_key}")
                    if target_key in master_data and 'nmr_data' in master_data[target_key]:
                        threshold_calc_data = {
                            'target_simulation': master_data[target_key]['nmr_data'],
                            'starting_material_simulations': [],
                            'calculation_timestamp': datetime.now().isoformat()
                        }
                        self.logger.debug(f"Found target simulation data: {json.dumps(threshold_calc_data['target_simulation'], indent=2)}")
                    else:
                        self.logger.warning(f"No NMR data found for target key {target_key}")
                        return {
                            'status': 'error',
                            'message': f"No NMR data found for target key {target_key}"
                        }
                    
                    # Get starting material simulation data from master data
                    for idx in range(len(starting_materials)):
                        start_key = f"{sample_id}_starting_{idx}"
                        if start_key in master_data and 'nmr_data' in master_data[start_key]:
                            threshold_calc_data['starting_material_simulations'].append(
                                master_data[start_key]['nmr_data']
                            )
                    
                    # Clean up temporary entries
                    keys_to_remove = [f"{sample_id}_target"] + [f"{sample_id}_starting_{i}" for i in range(len(starting_materials))]
                    for key in keys_to_remove:
                        master_data.pop(key, None)
                    
                    # Save master data once
                    with open(self.master_data_path, 'w') as f:
                        json.dump(master_data, f, indent=2)
                    
                finally:
                    # Clean up temporary file
                    if temp_json_path.exists():
                        temp_json_path.unlink()
                
                # Step 3: Run peak matching between target and starting material peaks
                self.logger.info("Starting peak matching process")
                self.logger.debug(f"Peak matching input - Target NMR: {json.dumps(threshold_calc_data['target_simulation'], indent=2)}")
                self.logger.debug(f"Peak matching input - Starting Material NMR: {json.dumps(threshold_calc_data['starting_material_simulations'][0], indent=2)}")
                
                # Get peaks from threshold calculation data
                target_peaks = threshold_calc_data.get('target_simulation', {})
                starting_material_peaks = threshold_calc_data.get('starting_material_simulations', [])
                
                # Check if we have valid peaks
                if not target_peaks:
                    return {
                        'status': 'error',
                        'message': 'No target peaks found for peak matching'
                    }
                if not starting_material_peaks:
                    return {
                        'status': 'error',
                        'message': 'No starting material peaks found for peak matching'
                    }

                def format_peaks(peaks, spectrum_type):
                    """Format peaks into the required structure for peak matching."""
                    if not peaks:  # Handle empty peaks list
                        if spectrum_type in ['1H', '13C']:
                            return {'shifts': [], 'Intensity': []}
                        else:  # 2D spectra (HSQC, COSY)
                            return {'F1 (ppm)': [], 'F2 (ppm)': [], 'Intensity': []}
                        
                    if spectrum_type == '1H':
                        # peaks is a list of [shift, intensity] lists
                        shifts = [float(p[0]) for p in peaks]
                        intensities = [float(p[1]) for p in peaks]
                        return {
                            'shifts': shifts,
                            'Intensity': intensities
                        }
                    elif spectrum_type == '13C':
                        # peaks is a list of shift values
                        shifts = [float(p) for p in peaks]
                        return {
                            'shifts': shifts,
                            'Intensity': [1.0] * len(shifts)
                        }
                    else:  # 2D spectra (HSQC, COSY)
                        # peaks is a list of [f1, f2] lists
                        f1_shifts = [float(p[0]) for p in peaks]
                        f2_shifts = [float(p[1]) for p in peaks]
                        return {
                            'F1 (ppm)': f1_shifts,
                            'F2 (ppm)': f2_shifts,
                            'Intensity': [1.0] * len(peaks)
                        }

                # Format NMR data for peak matching
                target_nmr = {}
                for spectrum_type, peaks in target_peaks.items():
                    if spectrum_type in ['1H_sim', '13C_sim', 'COSY_sim', 'HSQC_sim']:
                        target_nmr[spectrum_type[:-4]] = format_peaks(peaks, spectrum_type[:-4])

                starting_material_nmr = []
                for sm in starting_material_peaks:
                    sm_nmr = {}
                    for spectrum_type, peaks in sm.items():
                        if spectrum_type in ['1H_sim', '13C_sim', 'COSY_sim', 'HSQC_sim']:
                            sm_nmr[spectrum_type[:-4]] = format_peaks(peaks, spectrum_type[:-4])
                    starting_material_nmr.append(sm_nmr)

                # Extract specific NMR types
                self.logger.debug("Checking available spectra types")
                available_spectra = []
                for st in ['1H', '13C', 'HSQC', 'COSY']:
                    if st in target_nmr and st in starting_material_nmr[0]:
                        if st in ['1H', '13C']:
                            if len(target_nmr[st]['shifts']) > 0 and len(starting_material_nmr[0][st]['shifts']) > 0:
                                available_spectra.append(st)
                                self.logger.debug(f"Found valid {st} spectrum")
                        else:  # 2D spectra (HSQC, COSY)
                            if len(target_nmr[st]['F1 (ppm)']) > 0 and len(starting_material_nmr[0][st]['F1 (ppm)']) > 0:
                                available_spectra.append(st)
                                self.logger.debug(f"Found valid {st} spectrum")
                
                self.logger.info(f"Available spectra for peak matching: {available_spectra}")
                
                # Run peak matching between target and starting material peaks
                match_result = await self.peak_matching_tool.process(
                    {
                        'peaks1': target_nmr,  
                        'peaks2': starting_material_nmr[0]  # Taking only the first set
                    },
                    context={
                        'matching_mode': 'hung_dist_nn',
                        'error_type': 'sum',
                        'spectra': available_spectra
                    }
                )
                
                # Read results from the hardcoded location
                results_path = self.peak_matching_dir / 'current_run' / 'results.json'
                
                with open(results_path) as f:
                    peak_matching_results = json.load(f)
                
                if peak_matching_results['status'] == 'success':
                    # Format peak matching data for storage
                    self.logger.debug("Formatting peak matching results for storage")
                    peak_matching_data = {
                        'status': 'success',
                        'spectrum_errors': {},
                        'spectra': peak_matching_results['data']['results']
                    }
                    
                    # Extract spectrum errors
                    self.logger.debug("Extracting spectrum errors")
                    for spectrum_type, result in peak_matching_results['data']['results'].items():
                        if result['status'] == 'success':
                            error_val = result['overall_error']
                            peak_matching_data['spectrum_errors'][spectrum_type] = error_val
                            self.logger.debug(f"Error for {spectrum_type}: {error_val}")
                    
                    # Store peak matching results in master data for both target and starting material
                    all_data = self._load_molecular_data()
                    
                    # Store peak matching data for target molecule
                    if sample_id in all_data:
                        if 'exp_sim_peak_matching' not in all_data[sample_id]:
                            all_data[sample_id]['exp_sim_peak_matching'] = {}
                        all_data[sample_id]['exp_sim_peak_matching']['starting_material_comparison'] = peak_matching_data
                    
                    # Store peak matching data for starting material
                    for idx, sm_smiles in enumerate(starting_materials):
                        sm_id = f"{sample_id}_starting_{idx}"
                        if sm_id in all_data:
                            if 'exp_sim_peak_matching' not in all_data[sm_id]:
                                all_data[sm_id]['exp_sim_peak_matching'] = {}
                            all_data[sm_id]['exp_sim_peak_matching']['target_comparison'] = peak_matching_data
                    
                    # Save updated master data
                    self.logger.info("Saving updated master data with threshold results")
                    self._save_molecular_data(all_data)
                    
                    # Extract spectrum thresholds
                    spectrum_thresholds = {}
                    for spectrum_type, result in peak_matching_results['data']['results'].items():
                        if result['status'] == 'success':
                            spectrum_thresholds[spectrum_type] = result['overall_error']

                    # Calculate overall threshold
                    overall_threshold = sum(spectrum_thresholds.values()) / len(spectrum_thresholds)
                    self.logger.info(f"Calculated overall threshold: {overall_threshold}")
                    self.logger.debug(f"Individual spectrum thresholds: {json.dumps(spectrum_thresholds, indent=2)}")

                    # Extract just the matched_peaks from the results
                    simplified_results = {}
                    raw_results = peak_matching_results['data']['results']
                    for spectrum_type in raw_results:
                        simplified_results[spectrum_type] = {
                            'matched_peaks': raw_results[spectrum_type]['matched_peaks']
                        }

                    # Store threshold results in master data
                    threshold_data = {
                        'status': 'success',
                        'calculation_timestamp': datetime.now().isoformat(),
                        'type': 'peaks_vs_peaks',
                        'matching_mode': 'hung_dist_nn',
                        'error_type': 'sum',
                        'overall_threshold': overall_threshold,
                        'spectrum_thresholds': spectrum_thresholds,
                        'starting_material': starting_material_smiles,
                        'spectra': simplified_results  # Store only the matched peaks data
                    }
                    
                    all_data[sample_id]['threshold_data'] = threshold_data
                    self._save_molecular_data(all_data)
                    
                    return threshold_data
                    
                else:
                    return {
                        'status': 'error',
                        'message': f"Peak matching failed: {peak_matching_results.get('error', 'Unknown error')}"
                    }
                    
            except Exception as e:
                self.logger.error(f"Error in threshold calculation: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Threshold calculation failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}