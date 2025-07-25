"""
Tool for comparing spectral data between candidates and experimental data.
"""
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
from .data_extraction_tool import DataExtractionTool, DataSource
from .stout_operations import STOUTOperations
from .analysis_enums import DataSource

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectralComparisonTool:
    """Tool for analyzing and comparing spectral data between candidates."""
    
    def __init__(self, llm_service=None):
        """Initialize the spectral comparison tool."""
        self.llm_service = llm_service
        self.data_tool = DataExtractionTool()
        self.stout_ops = STOUTOperations()
        
    async def analyze_spectral_comparison(self, 
                                  workflow_data: Dict[str, Any], 
                                  context: Dict[str, Any],
                                  data_tool: Optional[DataExtractionTool] = None,
                                  spectral_tool: Optional['SpectralComparisonTool'] = None,
                                  llm_service: Optional[Any] = None) -> Dict[str, Any]:
        """
        Analyze spectral data across different types and candidates.
        
        Args:
            workflow_data: Dictionary containing molecular data and analysis results
            context: Additional context for the analysis, including:
                - num_candidates: Number of top candidates to analyze (default: 2)
            data_tool: Optional DataExtractionTool instance. If not provided, uses self.data_tool
            spectral_tool: Optional SpectralComparisonTool instance. If not provided, uses self
            llm_service: Optional LLM service instance. If not provided, uses self.llm_service
            
        Returns:
            Dictionary containing spectral comparison results
        """
        try:
            # Use provided tools/services or fallback to instance ones
            data_tool = data_tool or self.data_tool
            spectral_tool = spectral_tool 
            llm_service = llm_service or self.llm_service
            
            # Extract top candidates from previous analysis
            molecule_data = workflow_data["molecule_data"]
            # Get number of candidates from context, default to 2 if not specified
            num_candidates = context.get('num_candidates', 2)
            logger.info(f"Analyzing top {num_candidates} candidates from context")
            
            candidates = await self._get_top_candidates(molecule_data, num_candidates)
            if not candidates:
                return {
                    'type': 'error',
                    'content': 'No candidates found for spectral comparison'
                }
            
            # Analyze each candidate's spectral data
            candidate_analyses = []
            for candidate in candidates:
                analysis = await self._analyze_candidate_spectra(candidate, molecule_data)
                candidate_analyses.append(analysis)
            
            # Store results in intermediate file
            try:
                sample_id = molecule_data.get('sample_id')
                if sample_id:
                    # Load existing data first
                    try:
                        existing_data = await data_tool.load_data(sample_id, DataSource.INTERMEDIATE)
                    except FileNotFoundError:
                        existing_data = {
                            'analysis_results': {},
                            'completed_analysis_steps': {}
                        }

                    # Update with new analysis
                    existing_data['analysis_results']['spectral_analysis'] = {
                        'type': 'structure_peak_correlation',
                        'candidate_analyses': [analysis for analysis in candidate_analyses],
                    }
                    
                    # Mark spectral analysis as completed
                    existing_data['completed_analysis_steps']['spectral_analysis'] = True
                    
                    await data_tool.save_data(
                        existing_data, 
                        sample_id, 
                        DataSource.INTERMEDIATE
                    )
            except Exception as e:
                logger.error(f"Error saving spectral analysis to intermediate file: {str(e)}")
            
            # Add error handling for LLM service
            if all(not analysis.get('analysis_text') for analysis in candidate_analyses):
                logger.warning("No valid LLM analyses were generated. Check API key configuration.")
                
            return {
                'type': 'spectral_comparison',
                'content': {
                    'spectral_analysis': {
                        'candidate_analyses': [
                            {
                                'candidate_id': candidate.get('id'),
                                'smiles': candidate.get('smiles'),
                                'analysis': analysis
                            }
                            for candidate, analysis in zip(candidates, candidate_analyses)
                        ]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in spectral comparison: {str(e)}")
            return {
                'type': 'error',
                'content': str(e)
            }
            
    async def _get_top_candidates(self, molecule_data: Dict[str, Any], num_candidates: int = 2) -> List[Dict[str, Any]]:
        """Extract top candidates from previous analysis results.
        
        Args:
            molecule_data: Dictionary containing the sample_id and other molecular data
            num_candidates: Number of top candidates to return (default: 2)
            
        Returns:
            List of top N candidate molecules with their data
        """
        try:
            sample_id = molecule_data.get('sample_id')
            if not sample_id:
                logger.error("No sample_id found in molecule_data")
                return []
            
            # Try to load from intermediate file first, then master file
            for data_source in [DataSource.INTERMEDIATE, DataSource.MASTER_FILE]:
                logger.info(f"Attempting to load data from {data_source.value}")
                data = await self.data_tool.load_data(sample_id, data_source)
                
                if data and 'analysis_results' in data:
                    analysis_results = data['analysis_results']
                    logger.info(f"Found analysis_results with keys: {list(analysis_results.keys())}")
                    
                    if 'candidate_ranking' in analysis_results:
                        ranked_candidates = analysis_results['candidate_ranking'].get('ranked_candidates', [])
                        if ranked_candidates: 
                            logger.info(f"Found {len(ranked_candidates)} ranked candidates in {data_source.value}, returning top {num_candidates}")
                            return ranked_candidates[:num_candidates]  # Get top N candidates
            
            logger.warning(f"No ranked candidates found for sample {sample_id}")
            return []
            
        except Exception as e:
            logger.error(f"Error getting top candidates: {str(e)}")
            return []
            
    async def _analyze_candidate_spectra(self, candidate: Dict[str, Any], molecule_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze spectral data for a single candidate using the existing matching data.
        """
        try:
            # Extract the NMR data that already contains matching information
            nmr_data = candidate.get('nmr_data', {}).get('spectra', {})
            smiles = candidate.get('smiles')
            logger.info(f"Analyzing NMR data for candidate with SMILES: {smiles}")
            
            if not nmr_data:
                logger.warning(f"No NMR data found for candidate with SMILES: {smiles}")
                return {
                    'error': 'No NMR data available for analysis',
                    'candidate_id': candidate.get('rank'),
                    'smiles': smiles
                }
            
            # Analyze each spectrum type
            spectrum_analyses = {}
            valid_analyses = 0
            
            for spectrum_type in [ 'HSQC']: #['1H', '13C', 'HSQC', 'COSY']:
                if spectrum_type in nmr_data:
                    analysis = await self._analyze_matched_spectrum(
                        nmr_data[spectrum_type],
                        spectrum_type,
                        smiles,
                        candidate
                    )
                    # print("_____analysis")
                    # print(analysis)
                    # Check if analysis was successful
                    if analysis and analysis.get('structural_analysis', {}).get('analysis_text'):
                        valid_analyses += 1
                    # logger.info(f"___{spectrum_type} analysis: {analysis}")  # Comment out noisy logging
                    
                    spectrum_analyses[f"{spectrum_type}_analysis"] = analysis
            
            # # Log analysis results
            # if valid_analyses == 0:
            #     logger.warning(f"No valid spectral analyses generated for candidate with SMILES: {smiles}")
            # else:
            #     logger.info(f"Generated {valid_analyses} valid spectral analyses for candidate with SMILES: {smiles}")
            
            return {
                'candidate_id': candidate.get('rank'),  # Using rank as ID
                'smiles': smiles,
                'iupac_name': candidate.get('iupac_name'),
                'HSQC_score': candidate.get('scores', {}).get('HSQC'),
                'spectrum_analyses': spectrum_analyses,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing candidate spectra: {str(e)}")
            return {
                'error': str(e),
                'candidate_id': candidate.get('rank'),
                'smiles': candidate.get('smiles')
            }
            
    async def _analyze_matched_spectrum(self, spectrum_data: Dict[str, Any], spectrum_type: str, smiles: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze matched peaks between experimental and predicted spectra.
        
        Args:
            spectrum_data: Dictionary containing matched peak data
            spectrum_type: Type of NMR spectrum (1H, 13C, HSQC, COSY)
            smiles: SMILES string of the molecule
            candidate: Dictionary containing candidate molecule data including structure image
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # print("spectrum_data_____")
            # print(spectrum_data)
            # Extract matched peaks from spectrum data
            spectrum1 = spectrum_data.get('spectrum1', [])
            spectrum2 = spectrum_data.get('spectrum2', [])
            
            # Determine if this is a 2D spectrum
            is_2d = spectrum_type in ['HSQC', 'COSY']
            
            # Collect matching information
            peak_matches = []
            for peak1, peak2 in zip(spectrum1, spectrum2):
                if is_2d:
                    match_info = {
                        'experimental': {
                            'F1': peak1['F1 (ppm)'],  # Direct dictionary access
                            'F2': peak1['F2 (ppm)']
                        },
                        'predicted': {
                            'F1': peak2['F1 (ppm)'],
                            'F2': peak2['F2 (ppm)']
                        },
                        'error': peak1['Error'],
                        'atom_index': peak1['atom_index']
                    }
                else:
                    match_info = {
                        'experimental': {
                            'shift': peak1['shifts'],
                            'intensity': peak1['intensity']
                        },
                        'predicted': {
                            'shift': peak2['shifts'],
                            'intensity': peak2['intensity']
                        },
                        'error': peak1['Error'],
                        'atom_index': peak1['atom_index']
                    }
                peak_matches.append(match_info)
            
            # Get IUPAC name from intermediate data
            iupac_result = await self.stout_ops.get_iupac_name(smiles)

            # Format peak matches for prompt
            formatted_peaks = self._format_peak_matches(peak_matches, is_2d)
            
            # Create vision prompt
            prompt = f"""
First, I will provide a detailed structural description of the molecule shown in the image:
                    
1. Molecular Structure Description:
   - IUPAC Name: {iupac_result.get('iupac_name', 'Not available')}
   - Describe the complete molecular structure systematically, starting from a core feature
   - Note the connectivity and spatial arrangement of all atoms
   - Include atom numbering/labels as shown in the image
   - Identify key functional groups and structural motifs
   - Describe any notable stereochemistry or conformational features
   - Ensure the description is very detailed to allow reconstruction of the structure
   - Explain how the structure corresponds to its IUPAC name, particularly focusing on the parts relevant to the NMR analysis
   - Core structural features and key functional groups
   - Atom numbering as shown in the image

2. Evaluate how well this candidate molecule's simulated {spectrum_type} NMR spectrum matches the experimental data.

The image shows the proposed molecular structure with numbered atoms. These numbers correspond to the peak match data below, which compares experimental vs simulated chemical shifts.

SMILES: {smiles}

Peak Match Data (Experimental vs Simulated):
{formatted_peaks}

Please analyze and provide a detailed evaluation:

1. Structure Overview:
   - IUPAC Name: {iupac_result.get('iupac_name', 'Not available')}
   - Core structural features and key functional groups
   - Atom numbering as shown in the image
2. Region-by-Region Analysis:
   - Which regions show excellent agreement? 
   - Which regions show concerning deviations? 
   - Are the deviations systematic or random?
3. Peak Match Analysis:
   - Compare experimental vs predicted peaks systematically
   - Highlight matches with error < 0.1 as strong evidence
   - Flag deviations > 0.5 as concerning
   - Focus on patterns in the deviations and include their error numbers in the explanation
4. Chemical Environment Evaluation:
   - For well-matched peaks:
      * Confirm expected chemical shifts for the structural features
      * Note how they support the proposed structure
   - For significant deviations:
      * Analyze the chemical environment
      * Consider electronic and structural effects
      * Suggest possible explanations or alternatives
5. Structure Validation:
   - List the structural features confirmed by good matches
   - Identify substructures that need reconsideration
   - For problematic regions:
      * Quantify the spectral mismatch
      * Propose specific structural modifications or other explanations for the mismatch (e.g. impurities, solvents, noise, etc.)

Conclusion:
- Overall assessment of structure validity
- Confidence level based on spectral evidence
- Key recommendations for structural refinement

Remember to:
- Support conclusions with specific peak match data
- Focus on chemical reasoning for significant deviations
- Consider both confirming and contradicting evidence
- Follow each of the outlined analysis points 1-5 precisely
"""
            
            # logger.info("____prompt____")
            # logger.info(prompt)
            # Use vision capabilities for analysis
            try:
                analysis_result = await self.llm_service.analyze_with_vision(
                    prompt=prompt,
                    image_path=candidate['structure_image'],
                    model="claude-3-5-sonnet",  # Using better vision model
                    system=("You are an expert in NMR spectroscopy and structural analysis."
                           "Analyze how well the predicted NMR peaks match the experimental data, "
                           "focusing on structural features and chemical environments.")
                )
                
                # Ensure we have a valid analysis result
                if not isinstance(analysis_result, dict):
                    logger.warning(f"Invalid analysis result format: {type(analysis_result)}")
                    logger.warning(f"analysis_result: {analysis_result}")
                    analysis_result = {
                        'analysis_text': ''}
                
                return {
                    'type': spectrum_type,
                    'reasoning': analysis_result.get('analysis_text', ''),
                    'formatted_peaks': formatted_peaks,
                    'prompt': prompt,
                    'peak_matches': peak_matches,
                } 
                
            except Exception as e:
                logger.error(f"Error in vision analysis: {str(e)}")
                return {
                    'type': spectrum_type,
                    'reasoning': "",
                    'formatted_peaks': formatted_peaks,
                    'prompt': prompt,
                    'peak_matches': peak_matches,
                }
            
        except Exception as e:
            logger.error(f"Error analyzing matched spectrum {spectrum_type}: {str(e)}")
            return {}
            
    def _format_peak_matches(self, peak_matches: List[Dict[str, Any]], is_2d: bool) -> str:
        """Format peak matches in a chemically meaningful way."""
        formatted = []
        for match in peak_matches:
            exp = match['experimental']
            pred = match['predicted']
            error = match['error']
            atom_index = match['atom_index']

            if is_2d:
                # Format 2D NMR data
                formatted.append(
                    f"Atom {atom_index}: \n"
                    f"  Experimental: F1 δ {exp['F1']:.2f} ppm, F2 δ {exp['F2']:.2f} ppm\n"
                    f"  Predicted: F1 δ {pred['F1']:.2f} ppm, F2 δ {pred['F2']:.2f} ppm\n"
                    f"  Error: {error:.3f}"
                )
            else:
                # Format 1D NMR data
                formatted.append(
                    f"Atom {atom_index}: \n"
                    f"  Experimental: δ {exp['shift']:.2f} ppm, Intensity: {exp['intensity']:.2f}\n"
                    f"  Predicted: δ {pred['shift']:.2f} ppm, Intensity: {pred['intensity']:.2f}\n"
                    f"  Error: {error:.4f}"
                )
        return "\n".join(formatted)