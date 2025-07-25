"""
Tool for performing final comprehensive analysis of molecule candidates.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json
import ast
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import re
from .analysis_enums import DataSource, RankingMetric
from .data_extraction_tool import DataExtractionTool
import json as json_module  # Alias the json module to avoid name conflicts
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinalAnalysisTool:
    """Tool for performing final comprehensive analysis of molecule candidates."""
    
    def __init__(self, llm_service: Any = None):
        """Initialize the final analysis tool."""
        self.llm_service = llm_service
        self.data_tool = DataExtractionTool()
        
    async def analyze_final_results(self, workflow_data: Dict[str, Any], 
                                  context: Dict[str, Any],
                                  data_tool: Optional[DataExtractionTool] = None,
                                  llm_service: Optional[Any] = None) -> Dict[str, Any]:
        """
        Perform final comprehensive analysis of all available results.
        
        Args:
            data: Dictionary containing molecular data and analysis results
            context: Additional context including previous analysis results
            data_tool: Optional DataExtractionTool instance, uses self.data_tool if not provided
            llm_service: Optional LLM service instance, uses self.llm_service if not provided
            
        Returns:
            Dictionary containing final analysis results
        """
        try:
            logger.info("Starting final comprehensive analysis")
            
            # Initialize tools
            data, sample_id, molecule_data = await self._initialize_analysis(workflow_data, context, data_tool, llm_service)
            
            # Extract data using helper methods
            try:
                final_ranked_results = self._extract_ranked_candidates(data.get('analysis_results', {}))
                candidate_reasonings = self._extract_candidate_reasonings(data.get('analysis_results', {}))
                spectral_llm_reasonings = self._extract_spectral_llm_reasonings(data.get('analysis_results', {}))
            except Exception as e:
                logger.error(f"Error during data extraction: {str(e)}")
                raise

            # Get the overall LLM analysis
            llm_eval = data.get('analysis_results', {}).get('spectral_llm_evaluation', {})
            overall_llm_analysis = llm_eval.get('overall_analysis', 'No overall analysis available')
            logger.info(f"Found overall LLM analysis: {bool(overall_llm_analysis != 'No overall analysis available')}")

            # Generate comprehensive analysis using LLM
            if self.llm_service and final_ranked_results:
                logger.info("Starting comprehensive LLM analysis")
                
                # Prepare target molecule info
                target_info = {
                    'smiles': molecule_data.get('smiles'),
                    'molecular_weight': molecule_data.get('molecular_weight', ""),
                    'molecular_formula': molecule_data.get('molecular_formula', ""),
                    'experimental_data': molecule_data.get('experimental_data', {})
                }
                logger.info(f"Target molecule info prepared: {target_info}")
                
                # Generate prompt using helper methods
                try:
                    candidate_sections = self._generate_candidate_sections(final_ranked_results, candidate_reasonings)
                    analysis_prompt = self._generate_analysis_prompt(
                        target_info,
                        overall_llm_analysis,
                        spectral_llm_reasonings,
                        candidate_sections,
                        final_ranked_results
                    )
                    logger.info("Successfully generated analysis prompt")
                except Exception as e:
                    logger.error(f"Error generating prompt: {str(e)}")
                    raise
                
                try:
                    # Get model configurations
                    model_configs = self._get_model_configs()
                    
                    # Initialize variables to store responses
                    model_results = {}
                    
                    # Get analysis from each model
                    for model in [ 'deepseek']:# ['claude', 'deepseek', 'kimi', 'gemini', 'o3']:
                        raw_response, results, reasoning, thinking = await self._get_model_analysis(model, analysis_prompt, model_configs[model])
                        model_results[f'{model}_results'] = {
                            'raw_response': raw_response,
                            'content': results,
                            'reasoning_content': reasoning,
                            'thinking': thinking,
                            'analysis_prompt': analysis_prompt
                        }
                    logger.info("ölkj")
                    # Create final analysis output with all model data
                    final_analysis = self._create_final_analysis(sample_id, molecule_data, final_ranked_results, model_results, analysis_prompt, model_configs)
                    logger.info("asdfasd")

                    # Save final analysis to data
                    try:
                        # Create completed_analysis_steps if it doesn't exist
                        if 'completed_analysis_steps' not in data:
                            data['completed_analysis_steps'] = {}
                        
                        # Ensure analysis_results exists
                        if 'analysis_results' not in data:
                            data['analysis_results'] = {}
                        
                        # Store analysis results and mark as completed
                        data['analysis_results']['final_analysis'] = final_analysis
                        data['completed_analysis_steps']['final_analysis'] = True
                        
                        await self.data_tool.save_data(
                            data,
                            sample_id,
                            DataSource.INTERMEDIATE
                        )
                        logger.info(f"Successfully saved final analysis for sample {sample_id}")
                    except Exception as e:
                        logger.error(f"Error saving final analysis: {str(e)}")
                        raise
                    
                    return {
                        'type': 'success',
                        'content': final_analysis,
                    }
                
                except Exception as e:
                    logger.error(f"Error during LLM request: {str(e)}")
                    # Don't raise here, continue with what we have
                    print(final_ranked_results)
                    
                    # Ensure model_results has entries for all models with empty strings as defaults
                    default_model_result = {
                        'raw_response': '',
                        'content': {},
                        'reasoning_content': '',
                        'thinking': '',
                        'analysis_prompt': analysis_prompt or ''
                    }
                    
                    for model in ['claude', 'deepseek', 'kimi', 'gemini', 'o3']:
                        if f'{model}_results' not in model_results:
                            model_results[f'{model}_results'] = default_model_result.copy()
                        else:
                            # Ensure all fields exist in existing model results
                            current_result = model_results[f'{model}_results']
                            for key in default_model_result:
                                if key not in current_result:
                                    current_result[key] = default_model_result[key]
                    
                    # Prepare final output with whatever we have, ensuring all fields exist
                    final_analysis = {
                        'timestamp': datetime.now().isoformat(),
                        'sample_id': sample_id,
                        'target_info': {
                            'smiles': molecule_data.get('smiles', ''),
                            'molecular_weight': molecule_data.get('molecular_weight', '')
                        },
                        'analyzed_candidates': [
                            {
                                'smiles': result.get('smiles', ''),
                                'rank': result.get('rank', 0),
                                'molecular_weight': result.get('molecular_weight', ''),
                                'scores': result.get('scores', {}),
                                'confidence_score': result.get('confidence_score', 0.0),
                                'reasoning': result.get('reasoning', ''),
                                'llm_analysis': result.get('llm_analysis', {})
                            }
                            for result in final_ranked_results
                        ],
                        'llm_responses': {
                            'analysis_prompt': analysis_prompt or '',
                            'claude': {
                                'raw_response': model_results.get('claude_results', {}).get('raw_response', ''),
                                'reasoning_content': model_results.get('claude_results', {}).get('reasoning_content', ''),
                                'thinking': model_results.get('claude_results', {}).get('thinking', ''),
                                'parsed_results': model_results.get('claude_results', {}).get('content', {})
                            },
                            'deepseek': {
                                'raw_response': model_results.get('deepseek_results', {}).get('raw_response', ''),
                                'reasoning_content': model_results.get('deepseek_results', {}).get('reasoning_content', ''),
                                'thinking': model_results.get('deepseek_results', {}).get('thinking', ''),
                                'parsed_results': model_results.get('deepseek_results', {}).get('content', {})
                            },
                            'gemini': {
                                'raw_response': model_results.get('gemini_results', {}).get('raw_response', ''),
                                'reasoning_content': model_results.get('gemini_results', {}).get('reasoning_content', ''),
                                'thinking': model_results.get('gemini_results', {}).get('thinking', ''),
                                'parsed_results': model_results.get('gemini_results', {}).get('content', {})
                            },
                            'o3': {
                                'raw_response': model_results.get('o3_results', {}).get('raw_response', ''),
                                'reasoning_content': model_results.get('o3_results', {}).get('reasoning_content', ''),
                                'thinking': model_results.get('o3_results', {}).get('thinking', ''),
                                'parsed_results': model_results.get('o3_results', {}).get('content', {})
                            },
                            'kimi': {
                                'raw_response': model_results.get('kimi_results', {}).get('raw_response', ''),
                                'reasoning_content': model_results.get('kimi_results', {}).get('reasoning_content', ''),
                                'thinking': model_results.get('kimi_results', {}).get('thinking', ''),
                                'parsed_results': model_results.get('kimi_results', {}).get('content', {})
                            }
                        },
                        'metadata': {
                            'num_candidates': len(final_ranked_results),
                            'analysis_types_used': [],
                            'models_used': [],
                            'description': 'Comprehensive final analysis of all candidates based on available evidence',
                            'analysis_status': 'partial_failure',
                            'analysis_prompt': analysis_prompt or '',
                            'error_message': str(e)
                        }
                    }
                    # Save final analysis to data
                    try:
                        # Create completed_analysis_steps if it doesn't exist
                        if 'completed_analysis_steps' not in data:
                            data['completed_analysis_steps'] = {}
                        
                        # Ensure analysis_results exists
                        if 'analysis_results' not in data:
                            data['analysis_results'] = {}
                        
                        # Store analysis results and mark as completed
                        data['analysis_results']['final_analysis'] = final_analysis
                        data['completed_analysis_steps']['final_analysis'] = True
                        
                        await self.data_tool.save_data(
                            data,
                            sample_id,
                            DataSource.INTERMEDIATE
                        )
                        logger.info(f"Successfully saved final analysis for sample {sample_id}")
                    except Exception as e:
                        logger.error(f"Error saving final analysis: {str(e)}")
                        raise
                    
                    return {
                        'type': 'success',
                        'content': final_analysis,
                    }
                    
        except Exception as e:
            logger.error(f"Error in final analysis: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
            }

    async def analyze_with_deepseek(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Analyze using DeepSeek model via Azure endpoint."""
        try:
            thinking, content  = await self.llm_service.query_deepseek_azure(prompt, system_prompt)
            
            return {
                'model': 'deepseek-r1',
                'content': content,
                'thinking': thinking
            }
        except Exception as e:
            logger.error(f"Error in DeepSeek analysis: {str(e)}")
            return {
                'model': 'deepseek-r1',
                'content': str(e),
                'thinking': 'Error occurred during analysis'
            }

    def analyze_with_kimi(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Analyze using Kimi model."""
        try:
            thinking, content = self.llm_service._get_kimi_completion(
                message=prompt,
                model="kimi-thinking",
                max_tokens=8000,
                system=system_prompt
            )
            
            return {
                'model': 'kimi-thinking',
                'content': content,
                'thinking': thinking
            }
        except Exception as e:
            logger.error(f"Error in Kimi analysis: {str(e)}")
            return {
                'model': 'kimi-thinking',
                'content': str(e),
                'thinking': 'Error occurred during analysis'
            }

    async def _initialize_analysis(self, workflow_data: Dict[str, Any], context: Dict[str, Any],
                                 data_tool: Optional[DataExtractionTool] = None,
                                 llm_service: Optional[Any] = None) -> Tuple[Dict, str, Dict]:
        """Initialize analysis by setting up tools and extracting basic data."""
        data_tool = data_tool or self.data_tool
        if data_tool is None:
            raise ValueError("No data_tool provided and self.data_tool is None")
            
        llm_service = llm_service or self.llm_service
        if llm_service is None:
            raise ValueError("No llm_service provided and self.llm_service is None")

        # Extract molecule data and sample_id
        molecule_data = workflow_data.get('molecule_data', {})
        sample_id = (
            molecule_data.get('sample_id') or 
            workflow_data.get('sample_id') or 
            context.get('sample_id')
        )
        
        if not sample_id:
            raise ValueError("sample_id is required but not found in any data source")
            
        is_full_analysis = context.get('from_orchestrator', False)
        data_source = DataSource.INTERMEDIATE if is_full_analysis else DataSource.MASTER_FILE
        
        # Load data
        data = await data_tool.load_data(sample_id, data_source)
        
        return data, sample_id, molecule_data

    def _get_model_configs(self) -> Dict[str, Dict[str, str]]:
        """Get configuration for all LLM models."""
        base_system_prompt = ('You are an expert chemist specializing in structure elucidation and '
                            'spectral analysis. Analyze molecular candidates based on all available '
                            'evidence and provide detailed scientific assessments.')
        
        return {
            'claude': {'model': 'claude-3-5-sonnet', 'system': base_system_prompt},
            'deepseek': {'model': "DeepSeek-R1", 'system': base_system_prompt},
            'gemini': {'model': 'gemini-thinking', 'system': base_system_prompt},
            'o3': {'model': 'o3-mini', 'system': base_system_prompt},
            'kimi': {'model': 'kimi-thinking', 'system': base_system_prompt}
        }
        #'deepseek': {'model': 'deepseek-reasoner', 'system': base_system_prompt},

    async def _get_model_analysis(self, model: str, analysis_prompt: str, 
                                model_config: Dict[str, str]) -> Tuple[str, Dict, str, str]:
        """Get analysis from a specific model."""
        try:
            if model == 'deepseek':
                response_dict = await self.analyze_with_deepseek(analysis_prompt, model_config['system'])
                raw_response = response_dict.get('content', '')
                thinking = response_dict.get('thinking', '')
                results, reasoning = self._process_model_response(raw_response, model)
                logger.info(f"DeepSeek analysis (first 100 chars): {raw_response[:100]}")
                return raw_response, results, reasoning, thinking
            elif model == 'kimi':
                response_dict = self.analyze_with_kimi(analysis_prompt, model_config['system'])
                raw_response = response_dict.get('content', '')
                thinking = response_dict.get('thinking', '')
                results, reasoning = self._process_model_response(raw_response, model)
                logger.info(f"Kimi analysis (first 100 chars): {raw_response[:100]}")
                return raw_response, results, reasoning, thinking
            else:
                thinking = ""
                raw_response = await self.llm_service.get_completion(
                    message=analysis_prompt,
                    max_tokens=8000,
                    model=model_config['model'],
                    system=model_config['system']
                )
                results, reasoning = self._process_model_response(raw_response, model)
                logger.info(f"{model.capitalize()} analysis (first 100 chars): {raw_response[:100]}")
                return raw_response, results, reasoning, thinking
        except Exception as e:
            logger.error(f"Error getting {model} analysis: {str(e)}")
            return None, None, None, None

    def _create_candidate_analysis(self, candidate: Dict[str, Any], model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create analysis for a single candidate including all model results."""
        llm_analysis = {}
        for model in ['claude', 'deepseek', 'kimi', 'gemini', 'o3']:
            specific_model_results = model_results.get(f'{model}_results', {})
            model_content = specific_model_results.get('content', {})
            candidates = model_content.get('candidates', [])

            # Log candidate analysis
            logger.info(f"Analyzing candidate: {candidate.get('smiles', 'Unknown SMILES')}")
            logger.info(f"Candidate rank: {candidate.get('rank', 'Unknown')}")
            logger.info(f"Candidate confidence score: {candidate.get('confidence_score', 0.0)}")
            
            matching_candidate = next(
                (c for c in candidates if c.get('smiles') == candidate.get('smiles')), 
                {}
            )
            # logger.info(f"matching_candidate: {matching_candidate}")
            # logger.info(f"matching_candidate: {[c for c in candidates]}")

            llm_analysis[model] = {
                'confidence_score': matching_candidate.get('confidence_score'),
                'reasoning': matching_candidate.get('reasoning')
            }
            logger.info(f"llm_analysis: {llm_analysis}")

        # # Log LLM analysis results
        # for model, analysis in llm_analysis.items():
        #     logger.info(f"{model.capitalize()} confidence score: {analysis.get('confidence_score', 'N/A')}")
        #     logger.debug(f"{model.capitalize()} reasoning: {analysis.get('reasoning', 'N/A')[:100]}...")
            
        return {
            'smiles': candidate.get('smiles', ''),
            'rank': candidate.get('rank',"N/A"),
            'molecular_weight': candidate.get('molecular_weight', "N/A"),
            'scores': candidate.get('scores', {}),  
            'confidence_score': candidate.get('confidence_score', "N/A"),
            'reasoning': candidate.get('reasoning', ''),
            'llm_analysis': llm_analysis
        }

    def _create_final_analysis(self, sample_id: str, molecule_data: Dict[str, Any],
                             final_ranked_results: List[Dict], model_results: Dict[str, Dict],
                             analysis_prompt: str, model_configs: Dict[str, Dict]) -> Dict[str, Any]:
        """Create the final analysis dictionary with all results."""
        logger.info("-___________dsd_______________-")

        return {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'target_info': {
                'smiles': molecule_data.get('smiles'),
                'molecular_weight': molecule_data.get('molecular_weight')
            },
            'analyzed_candidates': [
                self._create_candidate_analysis(result, model_results)
                for result in final_ranked_results
            ],
            'llm_responses': {
                model: {
                    'raw_response': model_results.get(f'{model}_results', {}).get('raw_response'),
                    'parsed_results': model_results.get(f'{model}_results', {}).get('content'),
                    'thinking': model_results.get(f'{model}_results', {}).get('thinking'),
                    'reasoning_content': model_results.get(f'{model}_results', {}).get('reasoning_content'),
                    'analysis_prompt': model_results.get(f'{model}_results', {}).get('analysis_prompt'),
                    'config': model_configs[model]
                }
                for model in ['claude', 'deepseek', 'kimi', 'gemini', 'o3']
            },
            'metadata': {
                'num_candidates': len(final_ranked_results),
                'analysis_types_used': list(model_results.keys()),
                'models_used': [config['model'] for config in model_configs.values()],
                'description': 'Comprehensive final analysis of all candidates based on available evidence',
                'analysis_status': 'complete' if any(model_results.values()) else 'partial_failure',
            }
        }

    def _extract_ranked_candidates(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and format ranked candidates from analysis results.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            List of formatted candidate results
        """
        try:
            logger.info("Starting extraction of ranked candidates")
            final_results = []
            
            # First try spectral LLM evaluation results
            spectral_llm_eval = analysis_results.get('spectral_llm_evaluation', {})
            if not spectral_llm_eval:
                logger.warning("No spectral LLM evaluation results found")
                return []
                
            logger.info(f"Found spectral LLM evaluation with keys: {list(spectral_llm_eval.keys())}")
            candidates = spectral_llm_eval.get('candidates', [])
            logger.info(f"Found {len(candidates)} candidates in spectral LLM evaluation")
            
            # Log full candidate data for debugging
            for i, candidate in enumerate(candidates):
                # logger.info(f"Candidate {i} full data: {candidate}")
                logger.debug(f"Processing candidate with keys: {list(candidate.keys())}")
                smiles = candidate.get('smiles')
                if not smiles:
                    logger.warning("Candidate missing SMILES, skipping")
                    continue
                
                # Initialize molecular properties
                mol_weight = None
                formula = None
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        mol_weight = Descriptors.ExactMolWt(mol)
                        formula = rdMolDescriptors.CalcMolFormula(mol)
                except Exception as e:
                    logger.warning(f"Failed to calculate molecular properties for {smiles}: {str(e)}")
                
                result = {
                    'smiles': smiles,  
                    'rank': candidate.get('rank'),
                    'molecular_weight': mol_weight,
                    'formula': formula,
                    'scores': candidate.get('scores', {}),  
                    'spectral_analysis': {},
                    'confidence_score': 0.0,
                    'reasoning': candidate.get('reasoning', ''),
                    'iupac_name': candidate.get('iupac_name', 'Not available')
                }
                logger.info(f"Created result dict for candidate {i}: {result}")
                final_results.append(result)
                
            logger.info(f"Successfully extracted {len(final_results)} candidate results")
            # logger.info(f"Final results full data: {final_results}")
            return final_results
        except Exception as e:
            logger.error(f"Error extracting ranked candidates: {str(e)}")
            raise

    def _extract_candidate_reasonings(self, analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract spectral analysis reasonings for each candidate.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary mapping candidate IDs to their spectral reasonings
        """
        try:
            logger.info("Starting extraction of candidate reasonings")
            spectral_analysis = analysis_results.get('spectral_analysis', {})
            logger.info(f"Found spectral analysis with keys: {list(spectral_analysis.keys())}")
            
            candidate_analyses = spectral_analysis.get('candidate_analyses', {})
            candidate_reasonings = {}
            
            for analysis in candidate_analyses:
                candidate_id = analysis.get('candidate_id')
                smiles = analysis.get('smiles')
                if not candidate_id or not smiles:
                    logger.warning(f"Skipping analysis due to missing candidate_id or SMILES")
                    continue
                    
                candidate_data = candidate_reasonings[candidate_id] = {}
                logger.debug(f"Extracting reasonings for candidate {candidate_id}")
                
                spectrum_analyses = analysis.get('spectrum_analyses', {})
                candidate_data['spectral_reasonings'] = {
                    'HSQC': spectrum_analyses.get('HSQC_analysis', {}).get('reasoning', ''),
                    '1H': spectrum_analyses.get('1H_analysis', {}).get('reasoning', ''),
                    '13C': spectrum_analyses.get('13C_analysis', {}).get('reasoning', ''),
                    'COSY': spectrum_analyses.get('COSY_analysis', {}).get('reasoning', '')
                }
            
            logger.info(f"Successfully extracted reasonings for {len(candidate_reasonings)} candidates")
            return candidate_reasonings
        except Exception as e:
            logger.error(f"Error extracting candidate reasonings: {str(e)}")
            raise

    def _extract_spectral_llm_reasonings(self, analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract LLM reasonings for each spectrum type.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary mapping spectrum types to their LLM analysis
        """
        try:
            logger.info("Starting extraction of spectral LLM reasonings")
            spectral_llm_eval = analysis_results.get('spectral_llm_evaluation', {})
            logger.info(f"Found spectral LLM evaluation with keys: {list(spectral_llm_eval.keys())}")
            
            spectral_comparison = spectral_llm_eval.get('spectral_comparison', {})
            spectral_llm_reasonings = {}
            
            for spectrum_type in ['1H', '13C', 'HSQC', 'COSY']:
                spectrum_key = f"{spectrum_type}_exp"
                if spectrum_key in spectral_comparison:
                    spectrum_analyses = spectral_comparison[spectrum_key].get('analyses', [])
                    if spectrum_analyses:
                        spectral_llm_reasonings[spectrum_type] = {
                            'analysis_text': spectrum_analyses[0].get('analysis_text', ''),
                            'analysis_type': spectral_llm_eval.get('analysis_type', '')
                        }
                        logger.debug(f"Extracted LLM reasoning for {spectrum_type}")
                else:
                    logger.warning(f"No analysis found for spectrum type: {spectrum_type}")
            
            logger.info(f"Successfully extracted LLM reasonings for {len(spectral_llm_reasonings)} spectrum types")
            return spectral_llm_reasonings
        except Exception as e:
            logger.error(f"Error extracting spectral LLM reasonings: {str(e)}")
            raise

    def _generate_candidate_sections(self, final_results: List[Dict[str, Any]], 
                                   candidate_reasonings: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate detailed analysis sections for each candidate.
        
        Args:
            final_results: List of candidate results with scores and metadata
            candidate_reasonings: Dictionary mapping candidate IDs to their spectral reasonings
            
        Returns:
            List of formatted analysis sections for each candidate
        """
        try:
            logger.info(f"Starting generation of candidate analysis sections with {len(final_results)} results")
            # logger.info(f"Final results full data: {final_results}")
            logger.info(f"Candidate reasonings full data: {candidate_reasonings}")
            
            if not final_results:
                logger.error("No final results provided")
                return []
                
            candidate_sections = []
            
            for i, result in enumerate(final_results):
                logger.info(f"Processing result {i} with data: {result}")
                
                # Get candidate reasoning using index+1 (since candidate_reasonings starts at 1)
                candidate_data = candidate_reasonings.get(i + 1, {
                    'spectral_reasonings': {'HSQC': 'No analysis available'}
                })
                logger.info(f"Using reasoning for candidate {i + 1}")
                
                # Create analysis section
                try:
                    section = f""" 
                    Candidate {result.get('rank', 'Unknown')} Analysis:
                    IUPAC Name: {result.get('iupac_name', 'Not available')}
                    SMILES: {result.get('smiles', 'Not available')}
                    Molecular Weight: {result.get('molecular_weight', 'Not available')}
                    
                    Step 1. NMR Error Analysis:
                    - HSQC Error: {result.get('scores', {}).get('HSQC', 'N/A')}
                    
                    Step 2. Individual Spectral Analyses:
                    1. HSQC Analysis:
                    {candidate_data['spectral_reasonings'].get('HSQC', 'No analysis available')}
                    """
                    candidate_sections.append(section)
                    logger.info(f"Added analysis section for candidate {result.get('rank', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Error creating section for candidate {i}: {str(e)}")
                    continue
                
            logger.info(f"Generated {len(candidate_sections)} candidate analysis sections")
            return candidate_sections
        except Exception as e:
            logger.error(f"Error generating candidate sections: {str(e)}")
            raise

    def _generate_analysis_prompt(self, target_info: Dict[str, Any],
                                overall_llm_analysis: str,
                                spectral_llm_reasonings: Dict[str, Dict[str, str]],
                                candidate_sections: List[str],
                                final_ranked_results: List[Dict[str, Any]]) -> str:
        """
        Generate the complete analysis prompt for LLM processing.
        
        Args:
            target_info: Dictionary containing target molecule information
            overall_llm_analysis: Overall LLM analysis text
            spectral_llm_reasonings: Dictionary mapping spectrum types to their LLM analysis
            candidate_sections: List of formatted candidate analysis sections
            final_ranked_results: List of candidate results with scores and metadata
            
        Returns:
            Formatted analysis prompt string
        """
        try:
            logger.info("Generating complete analysis prompt")
            prompt = f"""
            You are tasked with making a final determination of the most likely correct molecular structure based on all available spectral and analytical evidence. Your analysis must be extremely thorough and systematic.
            
            Target Molecule Information:
            - Target Molecular Weight: {target_info.get('molecular_weight', 'Not available')}
            - Target Formula: {target_info.get('formula', 'Not available')}

            Candidate Information:
            {'\n'.join([f"Candidate {i+1}:"
                       f"\n- SMILES: {cand.get('smiles', 'Not available')}"
                       f"\n- Molecular Weight: {cand.get('molecular_weight', 'Not available')}"
                       f"\n- Formula: {cand.get('formula', 'Not available')}"
                       for i, cand in enumerate(final_ranked_results)])}
        
            Overall Spectral Analyses:
            1. HSQC Overall Analysis:
            {spectral_llm_reasonings.get('HSQC', {}).get('analysis_text', 'No analysis available')}
            Detailed Candidate Analyses:
            {'\n\n'.join(candidate_sections)}
            
            IMPORTANT: Provide a thorough analysis for EACH candidate structure in the processed list, followed by a clear final recommendation.
            Your response must end with a JSON result in the exact format shown below.
            Do not include any text after the JSON.
            For each candidate structure:
            1. Analyze all available spectral data
            2. Compare predicted vs experimental NMR shifts
            3. Evaluate structural features and their compatibility with data
            4. Consider molecular weight and other physical properties
            5. Assess data quality and potential issues
            
            Then synthesize all analyses to select the best candidate.
            
            CRITICAL: The end of your response MUST follow this EXACT JSON structure:

            JSON_RESULT = {{
                "candidates": [
                    {{
                        "smiles": "<SMILES string of this specific candidate>",
                        "confidence_score": <float between 0-1>,
                        "molecular_weight": <float>,
                        "reasoning": "Thorough evidence-based analysis for THIS SPECIFIC candidate addressing:
                                    - Detailed spectral analysis results
                                    - NMR shift comparisons and deviations
                                    - Structural feature evaluation
                                    - Molecular property matches/mismatches
                                    - Supporting and contradicting evidence
                                    Explain each point with specific data references.",
                        "data_quality_issues": {{
                            "title": "Brief description of quality concerns for this candidate",
                            "description": "Detailed explanation of ALL identified issues",
                            "impact": "high/medium/low",
                            "atom_index": <int between 0-50>
                        }}
                    }},
                    {{
                        "smiles": "<SMILES string of another candidate>",
                        "confidence_score": <float between 0-1>,
                        "molecular_weight": <float>,
                        "reasoning": "Thorough evidence-based analysis for THIS SPECIFIC candidate addressing:
                                    - Detailed spectral analysis results
                                    - NMR shift comparisons and deviations
                                    - Structural feature evaluation
                                    - Molecular property matches/mismatches
                                    - Supporting and contradicting evidence
                                    Explain each point with specific data references.",
                        "data_quality_issues": {{
                            "title": "Brief description of quality concerns for this candidate",
                            "description": "Detailed explanation of ALL identified issues",
                            "impact": "high/medium/low",
                            "atom_index": <int between 0-50>
                        }}
                    }}
                ],
                "final_recommendation": {{
                    "best_smiles": "<SMILES of the winning candidate>",
                    "overall_confidence": <float between 0-1>,
                    "molecular_weight_match": <boolean>,
                    "explanation": "Comprehensive justification for selecting this candidate:
                                  - Compare and contrast with other candidates
                                  - Highlight decisive factors in selection
                                  - Address any contradictions or uncertainties
                                  - Explain confidence level assessment
                                  - Discuss any remaining concerns"
                }}
            }}"""
                                    
            # 2. COSY Overall Analysis:
            # {spectral_llm_reasonings.get('COSY', {}).get('analysis_text', 'No analysis available')}
            
            # 3. 1H NMR Overall Analysis:
            # {spectral_llm_reasonings.get('1H', {}).get('analysis_text', 'No analysis available')}
            
            # 4. 13C NMR Overall Analysis:
            # {spectral_llm_reasonings.get('13C', {}).get('analysis_text', 'No analysis available')}

            logger.info("Successfully generated analysis prompt")
            return prompt
        except Exception as e:
            logger.error(f"Error generating analysis prompt: {str(e)}")
            raise

    def _process_model_response(self, raw_response: str, model_type: str) -> Tuple[Dict[str, Any], str]:
        """Process raw model response and extract both JSON result and reasoning."""
        try:
            result = self._extract_model_json_result(raw_response, model_type)
            return result['json_content'], result['reasoning_content']
        except Exception as e:
            logger.error(f"Error processing model response: {str(e)}")
            return {}, ""

    # def _extract_model_json_result(self, raw_text: str, model_type: str) -> Dict[str, Any]:
        #     """
    #     Extract JSON result from different model outputs based on their specific formats.

    #     Returns:
    #         Dictionary containing:
    #             - json_content: The parsed JSON content (or raw JSON substring if parsing fails)
    #             - reasoning_content: The reasoning text preceding the JSON
    #     """
    #     try:
    #         if not raw_text:
    #             logger.warning(f"Empty response from {model_type} model")
    #             return {'json_content': {}, 'reasoning_content': ''}

    #         if model_type == 'gemini':
    #             # Extract content between ```json and ```
    #             json_marker = "```json"
    #             reasoning_content = ''
    #             if json_marker in raw_text:
    #                 json_start = raw_text.find(json_marker)
    #                 reasoning_content = raw_text[:json_start].strip()
    #                 json_start += len(json_marker)
    #                 json_end = raw_text.find("```", json_start)
    #                 if json_end != -1:
    #                     json_content = raw_text[json_start:json_end].strip()
    #                     try:
    #                         return {
    #                             'json_content': json_module.loads(json_content),
    #                             'reasoning_content': reasoning_content
    #                         }
    #                     except (ValueError, SyntaxError) as e:
    #                         logger.warning(f"Failed to parse {model_type} JSON: {e}")
    #                         return {
    #                             'json_content': json_content,
    #                             'reasoning_content': reasoning_content
    #                         }
    #             return {'json_content': raw_text.strip(), 'reasoning_content': ''}

    #         elif model_type in ['o3', 'claude']:
    #             marker = "JSON_RESULT ="
    #             marker_index = raw_text.find(marker)
    #             if marker_index != -1:
    #                 # Everything before the marker is the reasoning content
    #                 reasoning_content = raw_text[:marker_index].strip()
    #                 remaining_text = raw_text[marker_index + len(marker):].strip()

    #                 # Find the beginning of the JSON object
    #                 start_brace_index = remaining_text.find("{")
    #                 if start_brace_index == -1:
    #                     logger.warning(f"No JSON object found in {model_type} response")
    #                     return {"reasoning_content": reasoning_content, "json_content": None}

    #                 # Use a counter to capture the complete JSON block (support nested braces)
    #                 brace_count = 0
    #                 end_index = None
    #                 for i, char in enumerate(remaining_text[start_brace_index:]):
    #                     if char == '{':
    #                         brace_count += 1
    #                     elif char == '}':
    #                         brace_count -= 1
    #                         if brace_count == 0:
    #                             end_index = start_brace_index + i + 1
    #                             break

    #                 if end_index is None:
    #                     logger.warning(f"No matching closing brace found in {model_type} response")
    #                     json_str = remaining_text[start_brace_index:]
    #                 else:
    #                     json_str = remaining_text[start_brace_index:end_index]

    #                 # Remove invalid control characters and clean up JSON string
    #                 json_str_clean = re.sub(r'[\x00-\x1f]+', " ", json_str)
                    
    #                 # First try to parse with json.loads after converting JSON literals
    #                 json_str_clean = (json_str_clean
    #                     .replace("True", "true")
    #                     .replace("False", "false")
    #                     .replace("None", "null")
    #                     .replace("'", '"'))
                    
    #                 try:
    #                     parsed_json = json.loads(json_str_clean)
    #                     logger.debug(f"Successfully parsed {model_type} JSON using json.loads")
    #                     return {
    #                         'json_content': parsed_json,
    #                         'reasoning_content': reasoning_content
    #                     }
    #                 except json.JSONDecodeError as e:
    #                     logger.warning(f"json.loads failed: {e}")
    #                     # Fallback: try ast.literal_eval with Python literals
    #                     python_literal = (json_str_clean
    #                         .replace("true", "True")
    #                         .replace("false", "False")
    #                         .replace("null", "None"))
    #                     try:
    #                         parsed_json = ast.literal_eval(python_literal)
    #                         logger.debug(f"Successfully parsed {model_type} JSON using ast.literal_eval")
    #                         return {
    #                             'json_content': parsed_json,
    #                             'reasoning_content': reasoning_content
    #                         }
    #                     except Exception as e2:
    #                         logger.warning(f"ast.literal_eval failed: {e2}")
    #                         return {
    #                             'json_content': json_str_clean,
    #                             'reasoning_content': reasoning_content
    #                         }
    #             else:
    #                 # Fallback: try a case-insensitive search for "json"
    #                 lower_text = raw_text.lower()
    #                 fallback_marker = "json"
    #                 fallback_index = lower_text.find(fallback_marker)
    #                 if fallback_index != -1:
    #                     reasoning_content = raw_text[:fallback_index].strip()
    #                     json_content = raw_text[fallback_index:].strip()
    #                     return {"reasoning_content": reasoning_content, "json_content": json_content}
    #                 else:
    #                     # If no marker is found, treat the entire text as reasoning
    #                     return {"reasoning_content": raw_text.strip(), "json_content": None}
    #         elif model_type == 'deepseek':
    #             try:
    #                 return {
    #                     'json_content': json_module.loads(raw_text),
    #                     'reasoning_content': ''
    #                 }
    #             except (ValueError, SyntaxError):
    #                 json_start = raw_text.find('{')
    #                 json_end = raw_text.rfind('}') + 1
    #                 if json_start != -1 and json_end > json_start:
    #                     reasoning_content = raw_text[:json_start].strip()
    #                     try:
    #                         return {
    #                             'json_content': json_module.loads(raw_text[json_start:json_end]),
    #                             'reasoning_content': reasoning_content
    #                         }
    #                     except (ValueError, SyntaxError) as e:
    #                         logger.warning(f"Failed to parse {model_type} JSON: {e}")
    #                         return {
    #                             'json_content': raw_text,
    #                             'reasoning_content': raw_text.strip()
    #                         }
    #                 return {'json_content': {}, 'reasoning_content': raw_text.strip()}

    #         logger.warning(f"No valid JSON format found for {model_type} model")
    #         return {'json_content': {}, 'reasoning_content': raw_text.strip()}

    #     except Exception as e:
    #         logger.warning(f"Failed to extract JSON for {model_type} model: {e}")
    #         return {'json_content': {}, 'reasoning_content': raw_text.strip()}

    def _extract_model_json_result(self, raw_text: str, model_type: str) -> Dict[str, Any]:
        """
        Extract JSON result from different model outputs based on their specific formats.
        
        Args:
            raw_text: Raw text output from the model
            model_type: Type of model ('gemini', 'claude', 'o3', 'deepseek')
            
        Returns:
            Dictionary containing:
                - json_content: The parsed JSON content (or raw JSON substring if parsing fails)
                - reasoning_content: The reasoning text preceding the JSON
        """
        if not raw_text:
            logger.warning(f"Empty response from {model_type} model")
            return {'json_content': {}, 'reasoning_content': ''}

        def clean_json_string(json_str: str) -> str:
            """Helper to clean and normalize JSON string."""
            # Remove invalid control characters
            cleaned = re.sub(r'[\x00-\x1f]+', " ", json_str)
            
            # Normalize boolean and null values
            cleaned = (cleaned
                    .replace("True", "true")
                    .replace("False", "false")
                    .replace("None", "null"))
                    
            # Remove any trailing commas before closing braces/brackets
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            return cleaned.strip()

        def find_json_boundaries(text: str) -> Tuple[int, int]:
            """Find the start and end indices of the outermost JSON object."""
            start = text.find('{')
            if start == -1:
                return -1, -1
                
            brace_count = 0
            end = -1
            
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
                        
            return start, end

        try:
            if model_type == 'gemini':
                json_marker = "```json"
                if json_marker in raw_text:
                    json_start = raw_text.find(json_marker) + len(json_marker)
                    json_end = raw_text.find("```", json_start)
                    
                    if json_end != -1:
                        reasoning_content = raw_text[:raw_text.find(json_marker)].strip()
                        json_str = raw_text[json_start:json_end].strip()
                        
                        try:
                            return {
                                'json_content': json.loads(clean_json_string(json_str)),
                                'reasoning_content': reasoning_content
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Gemini JSON: {e}")
                            return {
                                'json_content': json_str,
                                'reasoning_content': reasoning_content
                            }
                            
                # Try to find JSON without markers as fallback
                start, end = find_json_boundaries(raw_text)
                if start != -1 and end != -1:
                    return {
                        'json_content': raw_text[start:end],
                        'reasoning_content': raw_text[:start].strip()
                    }
                    
                return {'json_content': raw_text.strip(), 'reasoning_content': ''}

            elif model_type in ['o3', 'claude']:
                marker = "JSON_RESULT ="
                marker_index = raw_text.find(marker)
                
                if marker_index == -1:
                    # Try case-insensitive "json" as fallback
                    marker_index = raw_text.lower().find("json")
                    if marker_index != -1:
                        marker_len = 4  # len("json")
                    else:
                        # No marker found, look for raw JSON
                        start, end = find_json_boundaries(raw_text)
                        if start != -1 and end != -1:
                            return {
                                'json_content': raw_text[start:end],
                                'reasoning_content': raw_text[:start].strip()
                            }
                        return {'json_content': {}, 'reasoning_content': raw_text.strip()}
                else:
                    marker_len = len(marker)

                reasoning_content = raw_text[:marker_index].strip()
                json_text = raw_text[marker_index + marker_len:].strip()
                
                # Find and extract the JSON object
                start, end = find_json_boundaries(json_text)
                if start != -1 and end != -1:
                    json_str = clean_json_string(json_text[start:end])
                    
                    # Try parsing with json.loads
                    try:
                        return {
                            'json_content': json.loads(json_str),
                            'reasoning_content': reasoning_content
                        }
                    except json.JSONDecodeError:
                        # Try ast.literal_eval as fallback
                        try:
                            python_str = (json_str
                                .replace("true", "True")
                                .replace("false", "False")
                                .replace("null", "None"))
                            return {
                                'json_content': ast.literal_eval(python_str),
                                'reasoning_content': reasoning_content
                            }
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"Both parsing methods failed: {e}")
                            return {
                                'json_content': json_str,
                                'reasoning_content': reasoning_content
                            }
                
                return {'json_content': json_text, 'reasoning_content': reasoning_content}


            elif model_type in ['deepseek', 'kimi']:
                try:
                    return {
                        'json_content': json.loads(clean_json_string(raw_text)),
                        'reasoning_content': ''
                    }
                except json.JSONDecodeError:
                    start, end = find_json_boundaries(raw_text)
                    if start != -1 and end != -1:
                        json_str = clean_json_string(raw_text[start:end])
                        try:
                            return {
                                'json_content': json.loads(json_str),
                                'reasoning_content': raw_text[:start].strip()
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Deepseek JSON: {e}")
                            return {
                                'json_content': raw_text[start:end],
                                'reasoning_content': raw_text[:start].strip()
                            }
                    return {'json_content': {}, 'reasoning_content': raw_text.strip()}

            elif model_type == 'kimi':
                # Kimi uses OpenAI-style format, so we expect JSON content
                try:
                    return {
                        'json_content': json.loads(clean_json_string(raw_text)),
                        'reasoning_content': ''
                    }
                except json.JSONDecodeError:
                    # Try to find JSON object in the response
                    start, end = find_json_boundaries(raw_text)
                    if start != -1 and end != -1:
                        json_str = clean_json_string(raw_text[start:end])
                        try:
                            return {
                                'json_content': json.loads(json_str),
                                'reasoning_content': raw_text[:start].strip()
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Kimi JSON: {e}")
                            return {
                                'json_content': raw_text[start:end],
                                'reasoning_content': raw_text[:start].strip()
                            }
                    return {'json_content': {}, 'reasoning_content': raw_text.strip()}

            logger.warning(f"Unsupported model type: {model_type}")
            return {'json_content': {}, 'reasoning_content': raw_text.strip()}

        except Exception as e:
            logger.error(f"Unexpected error extracting JSON for {model_type} model: {e}")
            return {'json_content': {}, 'reasoning_content': raw_text.strip()}