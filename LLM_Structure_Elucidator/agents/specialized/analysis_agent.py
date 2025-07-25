"""
Agent for analyzing molecular structures and spectral data.
"""
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime

from ..base.base_agent import BaseAgent
from ..tools.analysis_enums import DataSource, RankingMetric
from ..tools.candidate_ranking_tool import CandidateRankingTool
from ..tools.structure_visualization_tool import StructureVisualizationTool
from ..tools.data_extraction_tool import DataExtractionTool
# from ..tools.molecular_visual_comparison_tool import MolecularVisualComparisonTool
from ..tools.spectral_comparison_tool import SpectralComparisonTool
from ..tools.final_analysis_tool import FinalAnalysisTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    # Rank and select top candidate molecules
    TOP_CANDIDATES = "top_candidates"
    # Compare predicted vs experimental spectral data
    SPECTRAL_COMPARISON = "spectral_comparison"
    # LLM-based evaluation of spectral matches
    SPECTRAL_LLM_EVALUATION = "spectral_llm_evaluation"  
    # Compare structural features between molecules
    STRUCTURAL_COMPARISON = "structural_comparison"
    # Identify and analyze key functional groups
    FUNCTIONAL_GROUP = "functional_group"
    # Validate NMR coupling patterns
    COUPLING_PATTERN = "coupling_pattern"
    # Calculate overall confidence scores
    CONFIDENCE_SCORING = "confidence_scoring"
    # Check for contradictions in analysis results
    CONTRADICTION_CHECK = "contradiction_check"
    # Final comprehensive analysis
    FINAL_ANALYSIS = "final_analysis"

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing molecular data using various analysis tools."""

    def __init__(self, llm_service: Optional[Any] = None):
        """Initialize the analysis agent."""
        capabilities = [
            "Top candidate selection and ranking",
            "Spectral analysis and comparison",
            "Structural comparison",
            "Functional group analysis",
            "Coupling pattern validation",
            "Confidence scoring",
            "Contradiction detection",
            "Final comprehensive analysis"
        ]
        super().__init__("Analysis Agent", capabilities)
        self.llm_service = llm_service
        self.data_tool = DataExtractionTool()
        self.ranking_tool = CandidateRankingTool(llm_service)
        self.structure_tool = StructureVisualizationTool()
        self.spectral_tool = SpectralComparisonTool(llm_service)
        self.final_tool = FinalAnalysisTool(llm_service)
        self.logger = logging.getLogger(__name__)

    def _create_temp_folder(self, sample_id: str) -> str:
        """Create a temporary folder for the current analysis run."""
        # Create base temp directory structure
        base_temp_dir = Path(__file__).resolve().parent.parent.parent / "_temp_folder"
        analysis_files_dir = base_temp_dir / "analysis_files"
        sample_temp_dir = analysis_files_dir / str(sample_id)
        
        # Create base temp folder and analysis_files if they don't exist
        base_temp_dir.mkdir(exist_ok=True)
        analysis_files_dir.mkdir(exist_ok=True)
        
        # Create a unique folder for this run using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_temp_dir = sample_temp_dir / timestamp
        run_temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created analysis directory at: {run_temp_dir}")
        return str(run_temp_dir)

    async def process(self, 
                    analysis_type: Union[AnalysisType, str],
                    workflow_data: Dict[str, Any],
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single analysis task based on the specified analysis type."""
        context = context or {}
        
        try:
            # Convert string to enum if needed
            if isinstance(analysis_type, str):
                analysis_type = AnalysisType(analysis_type)
                
            if analysis_type == AnalysisType.TOP_CANDIDATES:
                return await self._analyze_top_candidates(workflow_data, context)
            elif analysis_type == AnalysisType.SPECTRAL_COMPARISON:
                return await self._analyze_spectral_comparison(workflow_data, context)
            elif analysis_type == AnalysisType.SPECTRAL_LLM_EVALUATION:
                spectral_result = await self._analyze_spectral_llm_evaluation(workflow_data, context)
                # After spectral LLM evaluation, perform final analysis
            elif analysis_type == AnalysisType.FINAL_ANALYSIS:
                return await self._analyze_final_results(workflow_data, context)
               
            # elif analysis_type == AnalysisType.STRUCTURAL_COMPARISON:
            #     return await self._analyze_structural_comparison(workflow_data, context)
            # elif analysis_type == AnalysisType.FUNCTIONAL_GROUP:
            #     return await self._analyze_functional_groups(workflow_data, context)
            # elif analysis_type == AnalysisType.COUPLING_PATTERN:
            #     return await self._analyze_coupling_patterns(workflow_data, context)
            # elif analysis_type == AnalysisType.CONFIDENCE_SCORING:
            #     return await self._calculate_confidence_scores(workflow_data, context)
            # elif analysis_type == AnalysisType.CONTRADICTION_CHECK:
            #     return await self._check_contradictions(workflow_data, context)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Error in analysis process: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
                'metadata': {
                    'agent': 'ANALYSIS',
                    'confidence': 0.0,
                    'reasoning': 'Analysis process failed'
                }
            }

    async def process_all(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all analysis steps sequentially.
        
        Args:
            task_data: Dictionary containing task_input (with molecule_data and step_outputs) and context
        """
        try:
            # Add validation
            if not task_data.get('task_input'):
                raise KeyError("task_input missing from task_data")
            if not task_data['task_input'].get('workflow_data'):
                raise KeyError("workflow_data missing from task_input")
                
            workflow_data = task_data['task_input']["workflow_data"]
            if not workflow_data.get('molecule_data'):
                raise KeyError("molecule_data missing from workflow_data")        

            # Extract data from task input
            workflow_data = task_data['task_input']["workflow_data"]
            molecule_data = task_data['task_input']["workflow_data"]['molecule_data']
            context = task_data.get('context', {})
            # Log the result of each analysis step
            # logger.info(f"molecule_data {molecule_data}")
            # logger.info(f"context {context} ")
          
            # # Add step outputs to context if available
            # if 'step_outputs' in task_data['task_input']:
            #     context['step_outputs'] = task_data['task_input']['step_outputs']
            # logger.info(f"context {context} ")

            all_results = {}
            
            # Create temp folder for this analysis run
            sample_id = molecule_data.get('sample_id', 'unknown_sample')
            analysis_run_folder = self._create_temp_folder(sample_id)
            context['analysis_run_folder'] = analysis_run_folder
            context['from_orchestrator'] = True
            # logger.info(f"___context {context} ")

            # Sequential analysis pipeline
            analysis_steps = [
                (AnalysisType.TOP_CANDIDATES, self._analyze_top_candidates),
                (AnalysisType.SPECTRAL_COMPARISON, self._analyze_spectral_comparison),
                (AnalysisType.SPECTRAL_LLM_EVALUATION, self._analyze_spectral_llm_evaluation),
                (AnalysisType.FINAL_ANALYSIS, self._analyze_final_results)
                # (AnalysisType.STRUCTURAL_COMPARISON, self._analyze_structural_comparison),
                # (AnalysisType.FUNCTIONAL_GROUP, self._analyze_functional_groups),
                # (AnalysisType.COUPLING_PATTERN, self._analyze_coupling_patterns),
                # (AnalysisType.CONFIDENCE_SCORING, self._calculate_confidence_scores),
                # (AnalysisType.CONTRADICTION_CHECK, self._check_contradictions)
            ]

            for analysis_type, analysis_func in analysis_steps:
                try:
                    logger.info(f"Starting {analysis_type.value} analysis")
                    result = await analysis_func(workflow_data, context)
                    all_results[analysis_type.value] = result
                    
                    # Update context with results for next analysis
                    context['previous_analysis'] = context.get('previous_analysis', {})
                    context['previous_analysis'][analysis_type.value] = result
                    
                except Exception as e:
                    logger.error(f"Error in {analysis_type.value} analysis: {str(e)}")
                    all_results[analysis_type.value] = {
                        'status': 'error',
                        'error': str(e)
                    }

            return {
                'type': 'success',
                'content': all_results,
                'metadata': {
                    'agent': 'ANALYSIS',
                    'confidence': 1.0,
                    'reasoning': 'Completed all analysis steps'
                }
            }

        except Exception as e:
            logger.error(f"Error in process_all: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
                'metadata': {
                    'agent': 'ANALYSIS',
                    'confidence': 0.0,
                    'reasoning': 'Failed to complete analysis pipeline'
                }
            }


    async def _analyze_top_candidates(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze top candidate structures and prepare them for further analysis.
        """
        return await self.ranking_tool.analyze_top_candidates(
            workflow_data=workflow_data,
            context=context,
            data_tool=self.data_tool,
            ranking_tool=self.ranking_tool,
        )


    async def _analyze_spectral_comparison(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare spectral data across different types."""
        # Set default number of candidates if not in context
        if 'num_candidates' not in context:
            context['num_candidates'] = 3  # Default to 2 candidates
            logger.info(f"Setting default number of candidates to {context['num_candidates']}")
            
        return await self.spectral_tool.analyze_spectral_comparison(
            workflow_data=workflow_data,
            context=context,
            data_tool=self.data_tool,
            spectral_tool=self.spectral_tool,
            llm_service=self.llm_service
        )

    async def _analyze_spectral_llm_evaluation(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM-based evaluation of how well candidate structures match experimental NMR spectra.
        Uses vision capabilities to analyze structural features against spectral patterns.
        """
        try:
            self.logger.info("Starting spectral_llm_evaluation analysis")
            
            return await self.structure_tool.analyze_spectral_llm_evaluation(
                workflow_data=workflow_data,
                context=context,
                data_tool=self.data_tool,
                ranking_tool=self.ranking_tool,
                llm_service=self.llm_service
            )
        except Exception as e:
            self.logger.error(f"Error in spectral LLM evaluation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    # async def _analyze_structural_comparison(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Compare structural features between molecules."""
    #     try:
    #         # Initialize molecular comparison tool if not already present
    #         if not hasattr(self, 'molecular_comparison_tool'):
    #             self.molecular_comparison_tool = MolecularVisualComparisonTool()

    #         return await self.molecular_comparison_tool.analyze_structural_comparison(
    #             workflow_data=workflow_data,
    #             context=context
    #         )
    #     except Exception as e:
    #         logger.error(f"Error in structural comparison analysis: {str(e)}")
    #         return {
    #             'status': 'error',
    #             'error': str(e)
    #         }

    async def _analyze_final_results(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final analysis using the FinalAnalysisTool."""
        return await self.final_tool.analyze_final_results(
            workflow_data=workflow_data,
            context=context,
            data_tool=self.data_tool,
            llm_service=self.llm_service
        )
        
    # async def _analyze_functional_groups(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Analyze functional groups and their consistency."""
    #     # TODO: Implement functional group analysis
    #     pass

    # async def _analyze_coupling_patterns(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Analyze and validate coupling patterns."""
    #     # TODO: Implement coupling pattern analysis
    #     pass

    
    # async def _calculate_confidence_scores(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Calculate confidence scores for predictions."""
    #     # TODO: Implement confidence scoring
    #     pass

    # async def _check_contradictions(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Check for contradictions in spectral and structural data."""
    #     # TODO: Implement contradiction checking
    #     pass