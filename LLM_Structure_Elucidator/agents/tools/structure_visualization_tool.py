"""
Tool for generating and managing molecular structure visualizations.
"""
from __future__ import annotations 
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging
from rdkit import Chem
import os
import math
import numpy as np
from datetime import datetime
from .analysis_enums import DataSource, RankingMetric
from .data_extraction_tool import DataExtractionTool
from .stout_operations import STOUTOperations

# Set up logging
logger = logging.getLogger(__name__)

class StructureVisualizationTool:
    """Tool for generating and managing molecular structure visualizations."""

    def __init__(self):
        """Initialize the structure visualization tool."""
        self.logger = logger  # Use module-level logger
        self.stout_ops = STOUTOperations()

    def _format_experimental_data(self, exp_data: List, spectrum_type: str) -> str:
        """Format experimental NMR data based on spectrum type.
        
        Args:
            exp_data: List of experimental NMR data points
            spectrum_type: Type of NMR spectrum ('13C_exp', 'HSQC_exp', 'COSY_exp', or '1H_exp')
            
        Returns:
            Formatted string representation of the data
        """
        if not exp_data:
            return ""

        if spectrum_type == '13C_exp':  # 13C NMR has single values
            formatted_data = "13C NMR chemical shifts:\n"
            for shift in exp_data:
                formatted_data += f"{shift:.2f} ppm\n"
        elif spectrum_type in ['HSQC_exp', 'COSY_exp']:  # 2D NMR
            formatted_data = f"{spectrum_type} peaks (x, y coordinates):\n"
            for peak in exp_data:
                if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                    formatted_data += f"({peak[0]:.2f} ppm, {peak[1]:.2f} ppm)\n"
        else:  # 1H NMR
            formatted_data = "1H NMR peaks (shift, intensity):\n"
            for peak in exp_data:
                if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                    formatted_data += f"{peak[0]:.2f} ppm (intensity: {peak[1]:.1f})\n"
        
        return formatted_data

    async def _generate_individual_analysis_section(self, molecule_index: int, spectrum_type: str, formatted_data: str, smiles: str = None) -> str:
        """Generate analysis section for an individual molecule."""
        if spectrum_type != 'HSQC_exp':
            return ""  # Skip non-HSQC spectra
        
        iupac_name = "Not available"
        if smiles:
            iupac_result = await self.stout_ops.get_iupac_name(smiles)
            iupac_name = iupac_result.get('iupac_name', 'Not available')
            
        return f"""
                Molecule {molecule_index} Analysis:
                1. Structural Description:
                   - IUPAC Name: {iupac_name}
                   - Describe the overall molecular framework
                   - Identify key functional groups and their positions
                   - Note any distinctive structural features or patterns

                2. Expected HSQC Features:
                   - List the expected HSQC correlations based on structure
                   - Identify characteristic cross-peaks that should be present
                   - Note any unique HSQC patterns this structure should show

                3. Data Matching for Molecule {molecule_index}:
                   - Compare expected HSQC signals with experimental data:
                     {formatted_data}
                   - Identify which HSQC cross-peaks support this structure
                   - Note any missing or unexplained correlations
                """

    async def _generate_analysis_prompt(self, num_candidates: int, spectrum_type: str, formatted_data: str, smiles_list: List[str] = None) -> tuple[str, List[str]]:
        """Generate the complete analysis prompt for spectral evaluation.
        
        Args:
            num_candidates: Number of candidate molecules to analyze
            spectrum_type: Type of NMR spectrum being analyzed
            formatted_data: Formatted experimental data string
            smiles_list: Optional list of SMILES strings for the candidate molecules
            
        Returns:
            Tuple containing:
            - Complete analysis prompt string
            - List of IUPAC names for each molecule
        """
        # Generate individual analysis sections for each molecule
        individual_analysis_sections = []
        iupac_names = []
        for i in range(num_candidates):
            smiles = smiles_list[i] if smiles_list and i < len(smiles_list) else None
            section = await self._generate_individual_analysis_section(i+1, spectrum_type, formatted_data, smiles)
            individual_analysis_sections.append(section)
            
            # Get IUPAC name from the result
            if smiles:
                iupac_result = await self.stout_ops.get_iupac_name(smiles)
                iupac_names.append(iupac_result.get('iupac_name', 'Not available'))
            else:
                iupac_names.append('Not available')

        # Build the complete prompt
        prompt = f"""
                I'm showing you {num_candidates} candidate molecular structures (ranked left to right) and their experimental {spectrum_type} NMR data.

                Part 1: Individual Structure Analysis
                {' '.join(individual_analysis_sections)}

                Part 2: Comprehensive Comparative Analysis
                
                1. Detailed Structure Comparison:
                   a) Systematic Structural Analysis:
                      - For each molecule, provide a detailed breakdown of:
                         * Core scaffold identification and description
                         * Functional group positions and types
                         * Stereochemistry and conformational features
                      - Document exact atom indices for key features
                   
                   b) Comparative Feature Analysis:
                      - For each structural difference identified:
                         * Specify exact atom indices involved
                         * Describe the chemical environment changes
                         * Explain the potential impact on spectral properties
                      - Create a hierarchical list of differences, from most to least significant
                   
                   c) Common Elements Evaluation:
                      - Detail all shared structural motifs:
                         * Core frameworks
                         * Functional group patterns
                         * Stereochemical elements
                      - Explain how these commonalities support or challenge the structural assignments

                2. Evidence-Based Spectral Compatibility Analysis:
                   a) Detailed Ranking Justification:
                      - For each molecule, provide:
                         * Numerical score (1-10) for spectral match
                         * Specific peak assignments supporting the score
                         * Detailed explanation of any mismatches
                   
                   b) Critical Spectral Features:
                      - For each decisive spectral feature:
                         * Exact chemical shift values
                         * Coupling patterns and constants
                         * Correlation with structural elements
                         * Impact on structural validation
                   
                   c) Comparative Spectral Analysis:
                      - Create a feature-by-feature comparison:
                         * Chemical shift patterns
                         * Coupling relationships
                         * Through-space correlations
                      - Explain how each feature discriminates between candidates

                3. Comprehensive Confidence Evaluation:
                   a) Detailed Confidence Assessment:
                      - Provide a numerical confidence score (1-10)
                      - For each point affecting confidence:
                         * Specific evidence supporting the assessment
                         * Weight of the evidence (high/medium/low)
                         * Impact on overall structure determination
                   
                   b) Uncertainty Analysis:
                      - For each identified ambiguity:
                         * Exact location in the structure
                         * Nature of the uncertainty
                         * Impact on structure determination
                         * Potential alternative interpretations
                   
                   c) Data Gap Analysis:
                      - Identify missing experimental data:
                         * Specific experiments needed
                         * Expected information gain
                         * How it would resolve ambiguities
                      - Prioritize additional data needs

                Remember to:
                - Provide exact atom indices for all structural features discussed
                - Support each conclusion with specific spectral evidence
                - Quantify confidence levels for each assessment
                - Make explicit connections between structural features and spectral data
                - Present information in a clear, hierarchical format
                - Be thorough in documenting both supporting and contradicting evidence

                Part 3: Final Evaluation
                1. Structure Comparison:
                   - Compare key structural differences between all molecules
                   - Identify unique features in each candidate
                   - Note shared structural elements

                2. Spectral Compatibility Ranking:
                   - Rank structures from best to worst match with NMR data
                   - Provide specific evidence for each ranking
                   - Highlight decisive spectral features

                3. Confidence Assessment:
                   - Rate confidence (1-10) in your top choice
                   - Explain key factors in your decision
                   - Identify any remaining ambiguities
                   - Suggest additional data needed for confirmation
                """
        
        return prompt, iupac_names

    def _generate_overall_analysis_prompt(self, spectral_comparison: Dict[str, Any], top_candidates: List[Dict[str, Any]]) -> str:
        """Generate the complete overall analysis prompt focusing on HSQC analysis."""
        
        # Build individual molecule sections with HSQC analysis
        molecule_sections = []
        for i, candidate in enumerate(top_candidates):
            # Get HSQC scores and analysis from spectral comparison
            hsqc_data = None
            for spectrum_type, data in spectral_comparison.items():
                if spectrum_type == 'HSQC_exp' and 'candidates' in data:
                    hsqc_data = data['candidates'][i] if i < len(data['candidates']) else None
                    break
            
            section = f"""
            Molecule {i+1}:
            - IUPAC Name: {candidate.get('iupac_name', 'Not available')}
            - HSQC Score: {hsqc_data['score'] if hsqc_data and 'score' in hsqc_data else 'Not available'}
            - Structure Overview:
              * Describe the overall molecular framework
              * Note key functional groups and their positions
              * Highlight distinctive structural features that affect HSQC patterns
            """
            molecule_sections.append(section)

        return f"""
        Analyze the structural candidates based on their HSQC spectral matching, which is the most reliable indicator for structural matching in this analysis.

        Part 1: Individual Structure Analysis
        {' '.join(molecule_sections)}

        Part 2: HSQC-Based Analysis
        1. Structural Matching:
           - For each structure, evaluate:
             a) How well the structural features match the HSQC patterns (primary criterion)
             b) How well the predicted HSQC spectra match experimental data (shown by scores, lower is better)
           - Highlight any cases where HSQC scores contradict structural analysis
           - Explain how you resolved such contradictions, prioritizing logical structural consistency

        2. Confidence Assessment:
           - For each structure, provide:
             a) Final confidence score (1-10)
             b) Structural match confidence (how well structure explains HSQC patterns)
             c) HSQC match confidence (based on scores, lower is better)
           - Explain which factors were most influential in your scoring
           - Flag any concerning mismatches or contradictions

        3. Key Findings:
           - Identify the most diagnostic structural features supported by HSQC data
           - Highlight any discrepancies between structural analysis and HSQC scores
           - Discuss cases where:
             a) Low HSQC scores support structural analysis
             b) Low HSQC scores conflict with structural logic
             c) High HSQC scores might be acceptable due to strong structural evidence

        Remember: While HSQC scores provide valuable input (lower is better), they should not override clear structural evidence. 
        If a structure with slightly higher HSQC scores makes more chemical sense, this should be weighted more heavily 
        in the final analysis.
        """

    async def analyze_spectral_llm_evaluation(self,
                                            workflow_data: Dict[str, Any],
                                            context: Dict[str, Any],
                                            data_tool: DataExtractionTool,
                                            ranking_tool: 'CandidateRankingTool',
                                            llm_service: Any) -> Dict[str, Any]:
        """
        LLM-based evaluation of how well candidate structures match experimental NMR spectra.
        Uses vision capabilities to analyze structural features against spectral patterns.
        
        Args:
            workflow_data: Dictionary containing workflow data including molecule_data
            context: Context dictionary containing settings and state
            data_tool: Tool for extracting experimental data
            ranking_tool: Tool for ranking candidates
            llm_service: Service for LLM interactions
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Extract molecule_data from workflow_data
            logger.info(f"[analyze_spectral_llm_evaluation] workflow_data keys: {list(workflow_data.keys())}")
            if not workflow_data.get('molecule_data'):
                raise ValueError("No molecule_data found in workflow_data")
            
            molecule_data = workflow_data['molecule_data']
            sample_id = molecule_data['sample_id']

            # Determine data source and get ranking results
            # logger.info(f"[analyze_spectral_llm_evaluation] context: {context}")
            is_full_analysis = context.get('from_orchestrator', False)
            data_source = DataSource.INTERMEDIATE if is_full_analysis else DataSource.MASTER_FILE
            
            # Load data from appropriate source
            data = await data_tool.load_data(sample_id, data_source)
            # logger.info(f"[analyze_spectral_llm_evaluation] data keys: {list(data.keys())}")
            
            # Check if candidate ranking has been completed
            completed_steps = data.get('completed_analysis_steps', {})
            # logger.info(f"[analyze_spectral_llm_evaluation] completed_steps: {completed_steps}")
            candidate_ranking_completed = completed_steps.get('candidate_ranking', {})
            logger.info(f"[analyze_spectral_llm_evaluation] candidate_ranking_completed type: {type(candidate_ranking_completed)}, value: {candidate_ranking_completed}")
               
            if not candidate_ranking_completed:
                # Run candidate ranking if not completed
                # logger.info(f"Candidate ranking not found in {data_source.value}, running analysis...")
                candidate_ranking = await ranking_tool.analyze_candidates(
                        molecule_data=molecule_data,
                        sample_id=sample_id,
                        metric=RankingMetric.HSQC,  # Use HSQC for ranking
                        top_n=3,
                        include_reasoning=True
                    )
                
                # Reload data as it was updated by ranking tool
                data = await data_tool.load_data(sample_id, data_source)
            
            # Get ranking results which contain the image paths
            candidate_ranking = data.get('analysis_results', {}).get('candidate_ranking', {})
            if not candidate_ranking:
                raise ValueError("Candidate ranking results not found in data")
                
            # Extract image paths from ranking results
            mol_image_paths = []
            ranked_candidates = candidate_ranking.get('ranked_candidates', [])
            for candidate in ranked_candidates:
                image_path = candidate.get('structure_image')
                if not image_path:
                    raise ValueError(f"Structure image path not found for candidate rank {candidate.get('rank')}")
                mol_image_paths.append(image_path)
                
            # Get combined image path
            combined_image_path = candidate_ranking.get('combined_structure_image')
            if not combined_image_path:
                raise ValueError("Combined structure image path not found in ranking results")

            # Initialize results
            spectral_comparison = {}
            spectrum_types = ['HSQC_exp']
            # spectrum_types = ['HSQC_exp', 'COSY_exp', '1H_exp', '13C_exp']
            available_spectra = []

            # Check which spectra are available and get spectral comparison results
            for spectrum_type in spectrum_types:
                try:
                    exp_data = await data_tool.extract_experimental_nmr_data(
                        sample_id=molecule_data['sample_id'],
                        spectrum_type=spectrum_type,
                        source=DataSource.MASTER_FILE
                    )
                    if exp_data:
                        available_spectra.append(spectrum_type)
                except Exception as e:
                    logger.warning(f"Spectrum {spectrum_type} not available: {str(e)}")
                    continue

            if not available_spectra:
                raise ValueError("No experimental NMR data available for analysis")

            # Analyze each available spectrum type
            for spectrum_type in available_spectra:
                # Get experimental NMR data
                exp_data = await data_tool.extract_experimental_nmr_data(
                    sample_id=molecule_data['sample_id'],
                    spectrum_type=spectrum_type,
                    source=DataSource.MASTER_FILE
                )

                # Format the experimental data based on spectrum type
                formatted_data = self._format_experimental_data(exp_data, spectrum_type)
                if not formatted_data:
                    logger.warning(f"No data found for {spectrum_type}")
                    continue


                # Generate analysis prompt using helper functions
                analysis_prompt, iupac_names = await self._generate_analysis_prompt(
                    num_candidates=len(ranked_candidates),
                    spectrum_type=spectrum_type,
                    formatted_data=formatted_data,
                    smiles_list=[candidate['smiles'] for candidate in ranked_candidates]
                )

                # Get LLM analysis for this spectrum type (e.g., 3 times)
                analyses = []
                for _ in range(1): ########################### can run the analysis multiple times if needed
                    analysis = await llm_service.analyze_with_vision(
                        prompt=analysis_prompt,
                        image_path=str(combined_image_path),  # Pass single path string instead of list
                        model=context.get('model_choice', 'claude-3-5-sonnet')
                    )
                    analyses.append(analysis)
                spectral_comparison[spectrum_type] = {
                    'analyses': analyses,
                    'analysis_prompt': analysis_prompt,
                    'candidates': [{
                        'rank': candidate['rank'],
                        **{f"score_{spectrum_type.replace('_exp', '')}": candidate['scores'].get(spectrum_type.replace('_exp', ''))}
                    } for candidate in ranked_candidates],
                    'iupac_names': iupac_names
                }

            # # Final overall analysis comparing all spectrum types
            # overall_prompt = self._generate_overall_analysis_prompt(spectral_comparison, ranked_candidates)

            # # Use regular completion for final analysis (no vision needed)
            # overall_analysis = await llm_service.get_completion(
            #     message=overall_prompt,
            #     model=context.get('model_choice', 'claude-3-5-sonnet'),
            #     system="You are an expert in NMR spectroscopy analysis. Your task is to synthesize multiple spectral analyses into a clear, well-reasoned final evaluation."
            # )

            # Prepare final results
            evaluation_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'spectral_llm_evaluation',
                'spectral_comparison': spectral_comparison,
                # 'overall_analysis': overall_analysis,
                # 'overall_analysis_prompt': overall_prompt,
                'candidates': ranked_candidates
            }

            # Store results in appropriate file
            if is_full_analysis:
                data['analysis_results']['spectral_llm_evaluation'] = evaluation_results
                if 'completed_analysis_steps' not in data:
                    data['completed_analysis_steps'] = {}
                data['completed_analysis_steps']['spectral_llm_evaluation'] = True
                await data_tool.save_data(data, molecule_data['sample_id'], DataSource.INTERMEDIATE)
            else:
                master_data = await data_tool.load_data(molecule_data['sample_id'], DataSource.MASTER_FILE)
                master_data['analysis_results']['spectral_llm_evaluation'] = evaluation_results
                if 'completed_analysis_steps' not in master_data:
                    master_data['completed_analysis_steps'] = {}
                master_data['completed_analysis_steps']['spectral_llm_evaluation'] = True
                await data_tool.save_data(master_data, molecule_data['sample_id'], DataSource.MASTER_FILE)

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in spectral LLM evaluation: {str(e)}")
            raise

    def _create_temp_folder(self, sample_id: str) -> str:
        """Create a temporary folder for storing analysis files."""
        temp_folder = Path(f"temp/{sample_id}")
        temp_folder.mkdir(parents=True, exist_ok=True)
        return str(temp_folder)
