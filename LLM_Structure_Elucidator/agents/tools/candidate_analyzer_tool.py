"""Tool for analyzing candidate molecules from various prediction sources."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
from rdkit import Chem
import os
import pandas as pd
import ast

from .peak_matching_tool import EnhancedPeakMatchingTool
from .nmr_simulation_tool import NMRSimulationTool

class CandidateAnalyzerTool:
    """Tool for analyzing and scoring candidate molecules."""

    def __init__(self, analysis_type: str = None):
        """Initialize the candidate analyzer tool.
        
        Args:
            analysis_type: Type of analysis to perform ('forward', 'mol2mol', 'mmst', or None for all)
        """
        self.logger = logging.getLogger(__name__)
        self.peak_matcher = EnhancedPeakMatchingTool()
        self.nmr_tool = NMRSimulationTool()
        self.analysis_type = analysis_type
        
        # Set up paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.master_data_path = self.base_dir / "data" / "molecular_data" / "molecular_data.json"
        self.temp_dir = self.base_dir / "_temp_folder"
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        
        # Create all necessary directories at initialization
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(exist_ok=True)

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
        master_path = self.base_dir / "data" / "molecular_data" / "molecular_data.json"
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
        """Save data to intermediate file."""
        path = self._get_intermediate_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_molecular_data(self) -> Dict:
        """Load the molecular data from JSON file."""
        if self.master_data_path.exists():
            with open(self.master_data_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_molecular_data(self, data: Dict):
        """Save the molecular data to JSON file."""
        self.master_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.master_data_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_intermediate_results(self, sample_id: str, data: Dict[str, Any]) -> Path:
        """Save intermediate results for a sample to a temporary file.
        
        Args:
            sample_id: ID of the sample being processed
            data: Data to save for this sample
            
        Returns:
            Path to the saved intermediate file
        """
        # Create intermediate directory if it doesn't exist
        intermediate_dir = self.temp_dir / "intermediate_results"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Log directory creation status
        if intermediate_dir.exists():
            self.logger.info(f"Intermediate directory exists: {intermediate_dir}")
        else:
            self.logger.error(f"Failed to create intermediate directory: {intermediate_dir}")
            raise RuntimeError(f"Could not create intermediate directory: {intermediate_dir}")
    
        # Use a consistent filename for this sample
        intermediate_file = intermediate_dir / f"{sample_id}_intermediate.json"
        
        # Create parent directories for the file if needed
        intermediate_file.parent.mkdir(parents=True, exist_ok=True)
    
        # Save the data
        try:
            with open(intermediate_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Successfully saved intermediate results to: {intermediate_file}")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate results: {str(e)}")
            raise
        
        return intermediate_file

    def _combine_intermediate_results(self, intermediate_files: List[Path]) -> None:
        """Combine all intermediate results into the master data file.
        
        Args:
            intermediate_files: List of paths to intermediate result files
        """
        # Load current master data
        master_data = self._load_molecular_data()
        
        # Process each intermediate file
        for file_path in intermediate_files:
            with open(file_path, 'r') as f:
                sample_data = json.load(f)
                
            # Update master data with this sample's results
            for sample_id, data in sample_data.items():
                if sample_id not in master_data:
                    master_data[sample_id] = {}
                master_data[sample_id].update(data)
        
        # Save updated master data
        self._save_molecular_data(master_data)
        
        # Clean up intermediate files
        for file_path in intermediate_files:
            file_path.unlink()

    def _canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize SMILES string using RDKit.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonicalized SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Invalid SMILES string: {smiles}")
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            self.logger.error(f"Error canonicalizing SMILES {smiles}: {str(e)}")
            return None

    def _collect_unique_predictions(
        self,
        forward_predictions: List[Dict[str, Any]]
        ) -> Dict[str, Dict[str, Any]]:
        """Collect and deduplicate predictions based on canonicalized SMILES.
        
        Args:
            forward_predictions: List of prediction dictionaries
            
        Returns:
            Dictionary mapping canonicalized SMILES to their prediction info
        """
        unique_predictions: Dict[str, Dict[str, Any]] = {}
        
        # Sort predictions by log_likelihood to process most likely predictions first
        sorted_predictions = sorted(
            forward_predictions,
            key=lambda x: x.get("log_likelihood", float("-inf")),
            reverse=True
        )
        
        for prediction in sorted_predictions:
            if "all_predictions" not in prediction or not prediction["all_predictions"]:
                self.logger.warning("Skipping prediction with no all_predictions data")
                continue
                
            starting_material = prediction.get("starting_material", "")
            log_likelihood = prediction.get("log_likelihood", None)
            
            for smiles in prediction["all_predictions"]:
                canon_smiles = self._canonicalize_smiles(smiles)
                if not canon_smiles:
                    continue
                    
                # Only store the first (most likely) prediction for each unique molecule
                if canon_smiles not in unique_predictions:
                    unique_predictions[canon_smiles] = {
                        "starting_material": starting_material,
                        "log_likelihood": log_likelihood
                    }
                
        return unique_predictions

    async def _process_forward_synthesis(
        self,
        forward_predictions: List[Dict[str, Any]],
        sample_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process forward synthesis predictions."""
        self.logger.info(f"Starting forward synthesis processing with {len(forward_predictions)} predictions")
        self.logger.info(f"Sample data keys available: {list(sample_data.keys())}")
        
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        # First collect and deduplicate all predictions
        self.logger.info("Collecting and deduplicating predictions...")
        unique_predictions = self._collect_unique_predictions(forward_predictions)
        self.logger.info(f"Found {len(unique_predictions)} unique molecules after deduplication")
        
        # Try to get NMR predictions, but continue even if they fail
        nmr_predictions = {}
        try:
            self.logger.info("Starting batch NMR predictions...")
            nmr_predictions = await self._batch_process_nmr_predictions(
                    unique_predictions,
                    sample_data.get('sample_id', 'unknown')  # Pass sample_id here
                )            # self.logger.info(f"NMR predictions: {json.dumps(nmr_predictions, indent=2)}")

            self.logger.info(f"Completed NMR predictions for {len(nmr_predictions)} molecules")
        except Exception as e:
            self.logger.warning(f"NMR predictions failed: {str(e)}. Continuing without NMR data.")
        
        # Process each unique molecule
        self.logger.info("Starting analysis of individual molecules...")
        for canon_smiles, prediction_info in unique_predictions.items():
            self.logger.info(f"Analyzing molecule with SMILES: {canon_smiles}")
            molecule_result = await self._analyze_single_molecule(
                canon_smiles,  # Use canonicalized SMILES
                sample_data,
                source_info={
                    "source": "forward_synthesis",
                    "starting_material": prediction_info["starting_material"],
                    "log_likelihood": prediction_info["log_likelihood"]
                    },
                nmr_predictions=nmr_predictions.get(canon_smiles)  # Pass pre-computed predictions
                )
            if molecule_result:
                self.logger.info(f"Successfully analyzed molecule: {canon_smiles}")
                result["molecules"].append(molecule_result)
            else:
                self.logger.warning(f"Analysis failed for molecule: {canon_smiles}")

        self.logger.info(f"Processed {len(forward_predictions)} predictions into {len(result['molecules'])} unique molecules")
        return result

    async def _batch_process_nmr_predictions(
        self,
        unique_predictions: Dict[str, Dict[str, Any]],
        sample_id: str
        ) -> Dict[str, Dict[str, Any]]:
        """Process NMR predictions for a batch of unique molecules.
        
        Args:
            unique_predictions: Dictionary mapping canonicalized SMILES to their prediction info
            sample_id: ID of the sample being analyzed
            
        Returns:
            Dictionary mapping canonicalized SMILES to their NMR prediction data
        """
        predictions_data = {}
        failed_molecules = []
        
        try:
            # Create a temporary master JSON file for batch processing
            temp_master_data = {}
            prediction_keys = {}  # Map canonical SMILES to their temporary keys
            
            # Process each unique molecule with a properly structured sample ID
            for idx, (canon_smiles, prediction_info) in enumerate(unique_predictions.items(), start=1):
                try:
                    # Validate SMILES before adding to batch
                    if not self._canonicalize_smiles(canon_smiles):
                        self.logger.warning(f"Invalid SMILES string: {canon_smiles}")
                        failed_molecules.append((canon_smiles, "Invalid SMILES"))
                        continue
                        
                    # Create unique sample ID based on source and index
                    #source = prediction_info.get("starting_material", prediction_info.get("parent_smiles", "unknown"))
                    temp_key = f"pred_{idx}_{sample_id}"
                    
                    # Store mapping for later retrieval
                    prediction_keys[canon_smiles] = temp_key
                    
                    # Add molecule data with proper structure
                    temp_master_data[temp_key] = {
                        "smiles": canon_smiles,
                        "sample_id": temp_key
                    }
                except Exception as e:
                    self.logger.error(f"Error processing molecule {canon_smiles}: {str(e)}")
                    failed_molecules.append((canon_smiles, str(e)))
                    continue
            
            if not temp_master_data:
                self.logger.warning("No valid molecules to process")
                return predictions_data
            
            # Write temporary master data to file
            temp_master_path = self.temp_dir / "temp_master.json"

            with open(temp_master_path, "w") as f:
                json.dump(temp_master_data, f, indent=2)
            
            # Debug: Print the first few entries of temp_master_data
            temp_master_str = json.dumps(temp_master_data, indent=2)
            self.logger.info(f"First 500 characters of temp_master_data:\n{temp_master_str[:500]}")
            
            try:
                # Initialize NMR simulation tool
                nmr_tool = NMRSimulationTool()
                
                # Run batch NMR prediction
                # self.logger.info(f"Starting NMR prediction with {len(unique_predictions)} molecules")
                # self.logger.info(f"First 500 characters of temp_master_data:\n{temp_master_str[:500]}")
                # self.logger.info(f"Path of temp_master_path file: {temp_master_path}")

                result = await nmr_tool.simulate_batch(  
                    str(temp_master_path),
                    context={"use_slurm": False}
                )
                
                # self.logger.info(f"NMR prediction result: {result}")
                
                if result["status"] != "success":
                    self.logger.info(f"Batch NMR prediction failed: {result.get('message', 'Unknown error')}")
                    return predictions_data
                
                # Read the compiled results file
                result_file = result['data']['result_file']
                self.logger.info(f"Reading NMR predictions from {result_file}")
                try:
                    df = pd.read_csv(result_file)
                    # print("_---------------------------------------------------------")
                    # print(df)
                    # Process each molecule's predictions
                    for canon_smiles, temp_key in prediction_keys.items():
                        try:
                            # Find the row for this molecule
                            row = df[df['sample-id'] == temp_key]
                            if not row.empty:
                                # Extract NMR predictions
                                nmr_data = {}
                                
                                # Parse 1H NMR data
                                if '1H_NMR_sim' in row:
                                    try:
                                        peaks_str = row['1H_NMR_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['1H_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing 1H NMR data: {str(e)}")
                                        nmr_data['1H_sim'] = []

                                # Parse 13C NMR data
                                if '13C_NMR_sim' in row:
                                    try:
                                        peaks_str = row['13C_NMR_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['13C_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing 13C NMR data: {str(e)}")
                                        nmr_data['13C_sim'] = []

                                # Parse COSY data
                                if 'COSY_sim' in row:
                                    try:
                                        peaks_str = row['COSY_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['COSY_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing COSY data: {str(e)}")
                                        nmr_data['COSY_sim'] = []

                                # Parse HSQC data
                                if 'HSQC_sim' in row:
                                    try:
                                        peaks_str = row['HSQC_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['HSQC_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing HSQC data: {str(e)}")
                                        nmr_data['HSQC_sim'] = []
                                predictions_data[canon_smiles] = nmr_data
                                self.logger.info(f"Found NMR predictions for {canon_smiles}")
                            else:
                                self.logger.warning(f"No predictions found for {canon_smiles} in results file")
                                failed_molecules.append((canon_smiles, "No predictions in results file"))
                        except Exception as e:
                            self.logger.error(f"Error processing predictions for {canon_smiles}: {str(e)}")
                            failed_molecules.append((canon_smiles, str(e)))
                            
                except Exception as e:
                    self.logger.error(f"Error reading results file: {str(e)}")
                    return predictions_data
                
            finally:
                # Clean up temporary file
                if temp_master_path.exists():
                    temp_master_path.unlink()
            
        except Exception as e:
            self.logger.error(f"Error in batch NMR prediction: {str(e)}")
            # Don't raise - return partial results if any
        
        # Log summary of failures
        if failed_molecules:
            self.logger.warning(f"Failed to process {len(failed_molecules)} molecules:")
            for smiles, error in failed_molecules:
                self.logger.warning(f"  - {smiles}: {error}")
        
        self.logger.info(f"Successfully processed {len(predictions_data)} out of {len(unique_predictions)} molecules")
        return predictions_data

    async def _process_mol2mol(
        self,
        mol2mol_results: Dict[str, Any],
        sample_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process mol2mol predictions.
        
        Args:
            mol2mol_results: Dictionary containing mol2mol results with generated analogues
            sample_data: Sample data containing NMR experimental data and other metadata
        """
        self.logger.info("Starting mol2mol processing")
        self.logger.debug(f"Input mol2mol_results structure: {mol2mol_results}")
        self.logger.debug(f"Sample data keys: {list(sample_data.keys())}")
        
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        # First collect and deduplicate all predictions
        unique_predictions = {}
        generated_analogues = mol2mol_results.get("generated_analogues_target", {})
        self.logger.info(f"Found {len(generated_analogues)} target SMILES in generated_analogues_target")
        
        for target_smiles, analogues in generated_analogues.items():
            self.logger.debug(f"Processing analogues for target SMILES: {target_smiles}")
            self.logger.debug(f"Number of analogues: {len(analogues)}")
            
            for analogue_smiles in analogues:
                canon_smiles = self._canonicalize_smiles(analogue_smiles)
                if not canon_smiles:
                    self.logger.warning(f"Failed to canonicalize SMILES: {analogue_smiles}")
                    continue
                    
                if canon_smiles not in unique_predictions:
                    unique_predictions[canon_smiles] = {
                        "parent_smiles": target_smiles,
                        "source": "mol2mol"
                    }
        
        self.logger.info(f"Collected {len(unique_predictions)} unique predictions after canonicalization")
        
        # Batch process NMR predictions for all unique molecules
        self.logger.info("Starting NMR predictions batch processing")
        nmr_predictions = await self._batch_process_nmr_predictions(
                unique_predictions,
                sample_data.get('sample_id', 'unknown')
            )
        self.logger.info(f"Completed NMR predictions for {len(nmr_predictions)} molecules")

        # Process each unique molecule
        self.logger.info("Processing individual molecules")
        for canon_smiles, prediction_info in unique_predictions.items():
            self.logger.debug(f"Analyzing molecule: {canon_smiles}")
            molecule_result = await self._analyze_single_molecule(
                canon_smiles,
                sample_data,
                source_info=prediction_info,
                nmr_predictions=nmr_predictions.get(canon_smiles)
            )
            if molecule_result:
                result["molecules"].append(molecule_result)
            else:
                self.logger.warning(f"No result produced for molecule: {canon_smiles}")

        self.logger.info(f"Completed processing {len(unique_predictions)} unique mol2mol analogues")
        self.logger.info(f"Final result contains {len(result['molecules'])} molecules")
        return result

    async def _process_mmst(
        self,
        mmst_results: Dict[str, Any],
        sample_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process MMST predictions.
        
        Args:
            mmst_results: Dictionary containing MMST results with generated analogues
            sample_data: Sample data containing NMR experimental data and other metadata
        """
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        # Print MMST results
        # self.logger.info(f"MMST results: {json.dumps(mmst_results, indent=2)}")

        # First collect and deduplicate all predictions
        unique_predictions = {}
        for target_smiles, analogues in mmst_results.get("generated_analogues_target", {}).items():
            for analogue_smiles in analogues:
                canon_smiles = self._canonicalize_smiles(analogue_smiles)
                if not canon_smiles:
                    continue
                    
                if canon_smiles not in unique_predictions:
                    unique_predictions[canon_smiles] = {
                        "parent_smiles": target_smiles,
                        "source": "mmst"
                    }
        
        # Batch process NMR predictions for all unique molecules
        nmr_predictions = await self._batch_process_nmr_predictions(
                unique_predictions,
                sample_data.get('sample_id', 'unknown')
            )

        # Process each unique molecule
        for canon_smiles, prediction_info in unique_predictions.items():
            molecule_result = await self._analyze_single_molecule(
                canon_smiles,
                sample_data,
                source_info=prediction_info,
                nmr_predictions=nmr_predictions.get(canon_smiles)
            )
            if molecule_result:
                result["molecules"].append(molecule_result)

        self.logger.info(f"Processed {len(unique_predictions)} unique MMST predictions")
        return result


    async def _process_structure_generator(
        self,
        structure_predictions: List[Dict[str, Any]],
        experimental_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process structure generator predictions."""
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        for prediction in structure_predictions:
            molecule_result = await self._analyze_single_molecule(
                prediction["smiles"],
                experimental_data,
                source_info={
                    "method": prediction.get("method", "unknown"),
                    "starting_smiles": prediction.get("starting_point", "")
                }
            )
            if molecule_result:
                result["molecules"].append(molecule_result)

        return result

    async def _analyze_single_molecule(
        self,
        smiles: str,
        experimental_data: Dict[str, Any],
        source_info: Dict[str, Any],
        nmr_predictions: Optional[Dict[str, Any]] = None
        ) -> Optional[Dict[str, Any]]:
        """Analyze a single molecule against experimental data.
        
        Args:
            smiles: SMILES string of the molecule to analyze
            experimental_data: Dictionary containing experimental NMR data under 'nmr_data' key
            source_info: Information about how this molecule was generated
            nmr_predictions: Optional pre-computed NMR predictions for this molecule
        """
        # Get intermediate file path from experimental data
        # intermediate_file = Path(experimental_data.get("_intermediate_file_path"))
        # if not intermediate_file.exists():
        #     raise ValueError(f"Intermediate file not found: {intermediate_file}")

        try:
            self.logger.info(f"Starting analysis for molecule: {smiles}")
            self.logger.info(f"Available experimental data keys: {list(experimental_data.keys())}")
            self.logger.info(f"Source info: {source_info}")
             
            # Extract experimental NMR data
            nmr_data = experimental_data.get("nmr_data", {})
            if not nmr_data:
                self.logger.warning(f"No experimental NMR data found for molecule {smiles}")
                return None
                
            # Log available NMR data
            self.logger.info(f"Available experimental NMR data keys: {list(nmr_data.keys())}")
            if nmr_predictions:
                self.logger.info(f"Available predicted NMR data keys: {list(nmr_predictions.keys())}")
            
            experimental_spectra = {}
            predicted_spectra = {}
            
            # Format experimental peaks for each spectrum type
            for spectrum_type in ["1H", "13C", "HSQC", "COSY"]:
                exp_key = f"{spectrum_type}_exp"
                if exp_key in nmr_data and nmr_data[exp_key]:
                    try:
                        self.logger.info(f"Processing experimental {spectrum_type} peaks:")
                        # self.logger.info(f"Raw peaks: {nmr_data[exp_key]}")
                        
                        formatted_peaks = self._format_peaks(
                            nmr_data[exp_key], 
                            spectrum_type
                        )
                        if formatted_peaks:
                            experimental_spectra[spectrum_type] = formatted_peaks
                    except Exception as e:
                        self.logger.error(f"Error formatting experimental peaks for {spectrum_type}: {str(e)}")
                        continue

            # self.logger.info(f"nmr_predictions {nmr_predictions}")

            # Format predicted peaks for each spectrum type
            if nmr_predictions:
                for spectrum_type in ["1H", "13C", "HSQC", "COSY"]:
                    pred_key = f"{spectrum_type}_sim"  # Predicted data uses _sim suffix
                    # exp_key = f"{spectrum_type}_exp"  # Experimental data uses _exp suffix
                    
                    if pred_key in nmr_predictions and nmr_predictions[pred_key]:
                        try:
                            formatted_peaks = self._format_peaks(
                                nmr_predictions[pred_key],
                                spectrum_type
                            )
                            if formatted_peaks:
                                predicted_spectra[spectrum_type] = formatted_peaks
                        except Exception as e:
                            self.logger.error(f"Error formatting predicted peaks for {spectrum_type}: {str(e)}")
                            continue

            # Determine which spectra are available for matching
            available_spectra = []
            for st in ["1H", "13C", "HSQC", "COSY"]:
                # Both experimental and predicted data use base spectrum type
                if st in experimental_spectra and st in predicted_spectra:
                    # For 1D spectra (1H, 13C)
                    if st in ["1H", "13C"]:
                        if (len(experimental_spectra[st]['shifts']) > 0 and 
                            len(predicted_spectra[st]['shifts']) > 0):
                            available_spectra.append(st)
                    # For 2D spectra (HSQC, COSY)
                    else:
                        if (len(experimental_spectra[st]['F1 (ppm)']) > 0 and 
                            len(predicted_spectra[st]['F1 (ppm)']) > 0):
                            available_spectra.append(st)

            # Check if we have any spectra to analyze
            if not available_spectra:
                self.logger.error(f"No matching spectra available for analysis of {smiles}")
                self.logger.error(f"Experimental spectra keys: {list(experimental_spectra.keys())}")
                return None

            if not experimental_spectra or not predicted_spectra:
                self.logger.error(f"Missing experimental or predicted spectra for {smiles}")
                return None

            # Perform peak matching between experimental and predicted spectra
            peak_matching_result = await self.peak_matcher.process(
                {
                'peaks1': experimental_spectra,
                'peaks2': predicted_spectra
                },
                context={
                    'matching_mode': 'hung_dist_nn',
                    'error_type': 'sum',
                    'spectra': available_spectra
                }
                        )

            # self.logger.info(f"peak_matching_result {peak_matching_result}")
                        
            # After getting peak_matching_result
            if not peak_matching_result or 'status' not in peak_matching_result or peak_matching_result['status'] != 'success':
                self.logger.error(f"Peak matching returned invalid result for {smiles}")
                return None

            # Extract data from the result structure
            result_data = peak_matching_result.get('data', {})
            if not result_data or 'results' not in result_data:
                self.logger.error(f"Missing results data for {smiles}")
                return None

            # Process results for each spectrum type
            spectrum_errors = {}
            matched_peaks = {}

            for spectrum_type, spectrum_result in result_data['results'].items():
                if spectrum_result['status'] == 'success':
                    # Store the overall error for this spectrum
                    spectrum_errors[spectrum_type] = spectrum_result['overall_error']
                    
                    # Store the matched peaks
                    if 'matched_peaks' in spectrum_result:
                        matched_peaks[spectrum_type] = {
                            'spectrum1': spectrum_result['matched_peaks']['spectrum1'],
                            'spectrum2': spectrum_result['matched_peaks']['spectrum2'],
                            'spectrum1_orig': spectrum_result['original_data']['spectrum1'],
                            'spectrum2_orig': spectrum_result['original_data']['spectrum2']
                        }

            # Calculate overall score (lower error = better match)
            if spectrum_errors:
                overall_score = sum(spectrum_errors.values()) / len(spectrum_errors)
            else:
                self.logger.error(f"No spectrum errors found for {smiles}")
                return None

            results = {
                "smiles": smiles,
                "generation_info": source_info,
                "nmr_analysis": {
                    "spectra_matching": matched_peaks,
                    "matching_scores": {
                        "overall": overall_score,
                        "by_spectrum": spectrum_errors
                    }
                }
            }
  
            # In _analyze_single_molecule:
            # try:
            #     # Determine source type from source_info
            #     source_type = source_info.get("source")
            #     if source_type not in ["forward_synthesis", "mol2mol"]:
            #         raise ValueError(f"Invalid or missing source type: {source_type}. Must be either 'forward_synthesis' or 'mol2mol'")
                    
            #     if not sample_id:
            #         raise ValueError("Missing sample_id in experimental data")

            #     self.logger.error(f"experimental_data {experimental_data}")
            #     self._save_peak_matching_results(smiles, results, source_type, experimental_data.get("sample_id"), intermediate_file)
            # except Exception as e:
            #     self.logger.error(f"Error saving results for {smiles}: {str(e)}")

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing molecule {smiles}: {str(e)}")
            return None

    # def _save_peak_matching_results(self, smiles: str, results: Dict[str, Any], source_type: str, sample_id: str, intermediate_file: Path):
    #     """Save peak matching results for a molecule.
        
    #     Args:
    #         smiles: Canonicalized SMILES string of the molecule
    #         results: Dictionary containing peak matching results and analysis
    #         source_type: Type of analysis ('forward_synthesis' or 'mol2mol')
    #         sample_id: ID of the sample being analyzed
    #         intermediate_file: Path to the intermediate file containing sample data
    #     """
    #     # Load current sample data from intermediate file
    #     with open(intermediate_file, 'r') as f:
    #         current_data = json.load(f)
    #         sample_data = current_data[sample_id]
        
    #     # Initialize or get sample data structure
    #     if "candidate_analysis" not in sample_data:
    #         sample_data["candidate_analysis"] = {}
    #     if source_type not in sample_data["candidate_analysis"]:
    #         sample_data["candidate_analysis"][source_type] = {
    #             "tool_version": "1.0",
    #             "molecules": []
    #         }
        
    #     # Add timestamp to results
    #     results["timestamp"] = datetime.now().isoformat()
        
    #     # Get the correct section for this sample
    #     target_section = sample_data["candidate_analysis"][source_type]["molecules"]
        
    #     # Look for existing molecule entry
    #     molecule_entry = next(
    #         (mol for mol in target_section if mol.get("smiles") == smiles),
    #         None
    #     )
        
    #     if molecule_entry is None:
    #         # Create new molecule entry
    #         molecule_entry = {
    #             "smiles": smiles,
    #             "peak_matching_results": results
    #         }
    #         target_section.append(molecule_entry)
    #     else:
    #         # Update existing molecule entry
    #         molecule_entry["peak_matching_results"] = results
        
    #     # Save updated data back to intermediate file
    #     current_data[sample_id] = sample_data
    #     with open(intermediate_file, 'w') as f:
    #         json.dump(current_data, f, indent=2)
    
    #     self.logger.info(f"Saved peak matching results for molecule {smiles} in sample {sample_id} under {source_type}")

    def _format_peaks(self, peaks: List[Any], spectrum_type: str) -> Dict[str, Any]:
        """Format peaks into the required structure for peak matching.
        
        Args:
            peaks: List of peaks from NMR data
            spectrum_type: Type of spectrum ('1H', '13C', 'HSQC', 'COSY', or with _sim/_exp suffix)
             
        Returns:
            Dictionary with formatted peak data
        """
        # Extract base spectrum type by removing _sim or _exp suffix if present
        base_type = spectrum_type.split('_')[0]
        
        # self.logger.info(f"Formatting peaks for spectrum type: {spectrum_type} (base type: {base_type})")
        # self.logger.info(f"Input peaks: {peaks}")
        
        if not peaks:  # Handle empty peaks list
            if base_type in ['1H', '13C']:
                return {'shifts': [], 'Intensity': []}
            else:  # 2D spectra (HSQC, COSY)
                return {'F1 (ppm)': [], 'F2 (ppm)': [], 'Intensity': []}
            
        try:
            if base_type == '1H':
                # peaks is a list of [shift, intensity] lists
                shifts = [float(p[0]) for p in peaks]
                intensities = [float(p[1]) for p in peaks]
                formatted = {
                    'shifts': shifts,
                    'Intensity': intensities
                }
            elif base_type == '13C':
                # peaks is a list of shift values
                shifts = [float(p) for p in peaks]
                formatted = {
                    'shifts': shifts,
                    'Intensity': [1.0] * len(shifts)
                }
            else:  # 2D spectra (HSQC, COSY)
                # peaks is a list of [f1, f2] lists
                f1_shifts = [float(p[0]) for p in peaks]
                f2_shifts = [float(p[1]) for p in peaks]
                formatted = {
                    'F1 (ppm)': f1_shifts,
                    'F2 (ppm)': f2_shifts,
                    'Intensity': [1.0] * len(peaks)
                }
            
            #self.logger.info(f"Formatted peaks: {formatted}")
            return formatted
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error formatting peaks for {spectrum_type} spectrum: {str(e)}")
            self.logger.error(f"Problematic peaks data: {peaks}")
            if base_type in ['1H', '13C']:
                return {'shifts': [], 'Intensity': []}
            else:  # 2D spectra (HSQC, COSY)
                return {'F1 (ppm)': [], 'F2 (ppm)': [], 'Intensity': []}

    def _calculate_overall_score(self, peak_matching_result: Dict[str, Any]) -> float:
        """Calculate overall score from peak matching results."""
        # Implement scoring logic based on peak matching results
        # This is a placeholder - implement actual scoring logic
        return 0.0

    def _extract_spectrum_scores(self, peak_matching_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract individual spectrum scores from peak matching results."""
        # Implement logic to extract scores for each spectrum type
        # This is a placeholder - implement actual extraction logic
        return {
            "1H": 0.0,
            "13C": 0.0,
            "HSQC": 0.0,
            "COSY": 0.0
        }

    def _generate_summary(self, candidates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for all candidates."""
        summary = {
            "total_candidates": 0,
            "by_source": {},
            "top_candidates": []
        }

        # Calculate statistics
        for source, data in candidates.items():
            num_candidates = len(data.get("molecules", []))
            summary["by_source"][source] = num_candidates
            summary["total_candidates"] += num_candidates

        # Generate top candidates list (sorted by overall score)
        all_candidates = []
        for source, data in candidates.items():
            for molecule in data.get("molecules", []):
                all_candidates.append({
                    "smiles": molecule["smiles"],
                    "source": source,
                    "overall_score": molecule["nmr_analysis"]["matching_scores"]["overall"]
                })

        # Sort by overall score and take top N
        all_candidates.sort(key=lambda x: x["overall_score"], reverse=True)
        summary["top_candidates"] = all_candidates[:10]  # Top 10 candidates

        return summary

    async def process(
        self,
        molecular_data: Dict,
        context: Optional[Dict] = None
        ) -> Dict[str, Any]:
        """Process molecular data to analyze candidates.
        
        Args:
            molecular_data: Dictionary containing molecular data or sample_id
            context: Optional context dictionary
        """
        try:
            # Get sample_id from molecular_data or context
            sample_id = None
            if isinstance(molecular_data, dict):
                # Try to get sample_id directly from molecular_data
                sample_id = molecular_data.get('sample_id')
            
            if not sample_id and context and 'current_molecule' in context:
                sample_id = context['current_molecule'].get('sample_id')
            
            if not sample_id:
                self.logger.error("No sample_id found in molecular_data or context")
                return {
                    'status': 'error',
                    'error': 'No sample_id found in input data'
                }
                
            # Load intermediate data
            try:
                intermediate_data = self._load_or_create_intermediate(sample_id, context.get('current_molecule') if context else None)
                molecule_data = intermediate_data.get('molecule_data', {})
            except ValueError as e:
                self.logger.error(f"Error loading intermediate data: {str(e)}")
                return {
                    'status': 'error',
                    'error': f'Failed to load data: {str(e)}'
                }
            
            # Initialize result structure
            result = {
                "candidates": {},
                "sample_id": sample_id
            }
            
            try:
                # Process predictions based on analysis type
                if self.analysis_type == 'forward':
                    if "forward_predictions" in molecule_data:
                        self.logger.info(f"Processing forward synthesis predictions for {sample_id}")
                        self.logger.info(f"Number of predictions: {len(molecule_data['forward_predictions'])}")
                        
                        result["candidates"]["forward_synthesis"] = await self._process_forward_synthesis(
                            molecule_data["forward_predictions"],
                            molecule_data
                        )
                        
                        # Update and save results
                        if "candidate_analysis" not in molecule_data:
                            molecule_data["candidate_analysis"] = {}
                        molecule_data["candidate_analysis"]["forward_synthesis"] = result["candidates"]["forward_synthesis"]
                        
                elif self.analysis_type == 'mol2mol':
                    if "mol2mol_results" in molecule_data and molecule_data['mol2mol_results'].get('status') == 'success':
                        self.logger.info(f"Processing mol2mol predictions for {sample_id}")
                        
                        result["candidates"]["mol2mol"] = await self._process_mol2mol(
                            molecule_data["mol2mol_results"],
                            molecule_data
                        )
                        
                        # Update and save results
                        if "candidate_analysis" not in molecule_data:
                            molecule_data["candidate_analysis"] = {}
                        molecule_data["candidate_analysis"]["mol2mol"] = result["candidates"]["mol2mol"]
                        
                elif self.analysis_type == 'mmst':
                    if "mmst_results" in molecule_data and molecule_data['mmst_results'].get('status') == 'success':
                        self.logger.info(f"Processing MMST predictions for {sample_id}")
                        
                        result["candidates"]["mmst"] = await self._process_mmst(
                            molecule_data["mmst_results"],
                            molecule_data
                        )
                        
                        # Update and save results
                        if "candidate_analysis" not in molecule_data:
                            molecule_data["candidate_analysis"] = {}
                        molecule_data["candidate_analysis"]["mmst"] = result["candidates"]["mmst"]
                
                # Save updated data if any changes were made
                if result["candidates"]:
                    intermediate_data['molecule_data'] = molecule_data
                    self._save_intermediate(sample_id, intermediate_data)
                    
                    # Generate summary
                    result["summary"] = self._generate_summary(result["candidates"])
                    result["status"] = "success"
                else:
                    result["status"] = "no_candidates"
                    result["message"] = f"No candidates to analyze for {self.analysis_type} prediction type"
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error in candidate analysis: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        except Exception as e:
            self.logger.error(f"main: {str(e)}")
            return {
            "status": "error",
            "error": str(e)
                }
