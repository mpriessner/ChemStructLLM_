"""
Orchestration Agent for managing structure elucidation workflows.

This module implements the high-level orchestration logic for analyzing chemical structure data
and coordinating analysis workflows through the Coordinator Agent using LLM-generated commands.
"""
 
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import traceback
import uuid

from services.llm_service import LLMService
from ..coordinator.coordinator import CoordinatorAgent
from ..base.base_agent import BaseAgent
from .workflow_definitions import determine_workflow_type, get_workflow_steps, WorkflowType, WorkflowStep

class OrchestrationAgent(BaseAgent):
    """Agent responsible for orchestrating the structure elucidation workflow."""

    def __init__(self, llm_service: LLMService, coordinator=None):
        """Initialize orchestrator with required services."""
        capabilities = [
            "Workflow generation",
            "Process coordination",
            "Tool execution",
            "Error handling"
        ]
        super().__init__("Orchestration Agent", capabilities)
        self.llm_service = llm_service
        self.coordinator = coordinator
        self.tool_agent = coordinator.tool_agent if coordinator else None
        self.logger = self._setup_logger()

        # Set path to molecular data file using relative path from orchestrator location
        self.molecular_data_file = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
        
        # Create log directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent / 'temp' / 'memory' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_molecule_data(self, sample_id: str, initial_data: Dict = None) -> Dict:
        """Load fresh molecule data from molecular data file for a specific sample.
        
        Args:
            sample_id: The ID of the sample to load
            initial_data: Initial molecule data to store if molecular data file doesn't exist
        """
        try:
            if not self.molecular_data_file.exists():
                self.logger.error(f"Molecular data file not found at {self.molecular_data_file}")
                return None
                
            with open(self.molecular_data_file, 'r') as f:
                molecular_data = json.load(f)
                
            # Get molecule data directly using sample_id as key
            if sample_id in molecular_data:
                self.logger.info(f"[load_molecule_data] Successfully loaded fresh data for sample {sample_id}")
                return molecular_data[sample_id]
                    
            self.logger.error(f"[load_molecule_data] No data found for sample {sample_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"[load_molecule_data] Error loading molecule data: {str(e)}")
            return None

    async def process_molecule(self, molecule_data: Dict, user_input: str = None, context: Dict = None) -> Dict:
        """Process a single molecule through the structure elucidation workflow."""
        try:
            if not self.tool_agent:
                raise RuntimeError("Tool agent not initialized")
                
            # Use workflow type from context
            workflow_type = WorkflowType(context.get('workflow_type'))
            workflow_steps = get_workflow_steps(workflow_type)
            self.logger.info(f"[process_molecule] Using {len(workflow_steps)} workflow steps for workflow type: {workflow_type.value}")
            
            # Initialize context if not provided
            context = context or {}
            
            # Create temporary directory for this run
            run_id = str(uuid.uuid4())
            run_dir = Path("_temp_folder") / "structure_elucidation" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            context['run_dir'] = str(run_dir)
            context['run_id'] = run_id
            
            # Get initial sample_id
            sample_id = molecule_data.get('sample_id')
            if not sample_id:
                raise ValueError("No sample_id found in molecule_data")
            
            # Track workflow results and step completion
            workflow_data = {
                'predictions': [],
                'matches': [],
                'plots': [],
                'completed_steps': {},
                'step_outputs': {},
                'candidate_analysis': {}  # Add new key for combined candidate analysis
            }
            
            # Execute each step in sequence
            for idx, step in enumerate(workflow_steps, 1):
                try:
                    self.logger.info(f"[process_molecule] Step {idx}/{len(workflow_steps)}: {step.description}")
                    
                    # Load fresh molecule data before each step
                    fresh_molecule_data = self.load_molecule_data(sample_id)
                    if fresh_molecule_data is None:
                        error_msg = f"Failed to load fresh molecule data for step {idx}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Failed to load fresh molecule data'
                            }
                        }
                    
                    # Update context with fresh molecule data and any existing candidate analysis
                    context['current_molecule'] = fresh_molecule_data
                    if 'candidate_analysis' in fresh_molecule_data:
                        workflow_data['candidate_analysis'].update(fresh_molecule_data['candidate_analysis'])
                    
                    # Validate prerequisites based on step requirements
                    if not self._validate_step_prerequisites(step, workflow_data):
                        error_msg = f"Prerequisites not met for step {idx}: {step.description}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step prerequisites not met'
                            }
                        }
                    
                    # Add current step to context
                    context['current_step'] = step
                    
                    # Execute the step command
                    result = await self.tool_agent.process(step.command, context=context)
                    
                    # Validate step output
                    if not self._validate_step_output(step, result):
                        error_msg = f"Step {idx} failed validation: {step.description}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step output validation failed'
                            }
                        }
                    
                    # Store step results
                    if result.get('type') == 'success':
                        self.logger.info(f"[process_molecule] Step {idx} completed successfully")
                        workflow_data['completed_steps'][step.keyword] = True
                        workflow_data['step_outputs'][step.keyword] = result.get('content', {})
                        
                        # Handle molecular data updates from tools
                        if 'molecular_data' in result.get('content', {}):
                            self.logger.info(f"[process_molecule] Updating molecular data from tool results")
                            # Update the master molecular data file
                            with open(self.molecular_data_file, 'r') as f:
                                master_data = json.load(f)
                            # self.logger.info(f"[process_molecule] Tool results: {result}")
                            # self.logger.info(f"[process_molecule] Content: {result.get('content', {})}")
                            self.logger.info(f"[process_molecule] Molecular data: {result.get('content', {}).get('molecular_data', {})}")
                            
                            master_data.update(result['content']['molecular_data'])
                            with open(self.molecular_data_file, 'w') as f:
                                json.dump(master_data, f, indent=2)
                            self.logger.info(f"[process_molecule] Updated master molecular data file {master_data}")
                    else:
                        error_msg = f"Step {idx} returned error: {result.get('content', 'Unknown error')}"
                        self.logger.info(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step execution failed'
                            }
                        }
                    
                    # Update context with latest results
                    context['workflow_data'] = workflow_data
                    
                except Exception as e:
                    self.logger.error(f"[process_molecule] Error executing step {idx} '{step.description}': {str(e)}")
                    return {
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'ORCHESTRATOR',
                            'confidence': 0.0,
                            'reasoning': f'Failed to execute step {idx}: {step.description}'
                        }
                    }
            
            # Log final workflow statistics
            self.logger.info(f"[process_molecule] Workflow completed successfully:")
            self.logger.info(f"[process_molecule] - Completed steps: {len(workflow_data['completed_steps'])}/{len(workflow_steps)}")
            self.logger.info(f"[process_molecule] - Total predictions: {len(workflow_data['predictions'])}")
            self.logger.info(f"[process_molecule] - Total matches: {len(workflow_data['matches'])}")
            self.logger.info(f"[process_molecule] - Total plots: {len(workflow_data['plots'])}")
            
            # Clean up intermediate files
            # try:
            #     import shutil
            #     intermediate_results_dir = Path("_temp_folder") / "intermediate_results"
            #     if intermediate_results_dir.exists():
            #         shutil.rmtree(intermediate_results_dir)
            #         self.logger.info(f"[process_molecule] Cleaned up intermediate results directory: {intermediate_results_dir}")
                
            #     # Also clean up the run directory
            #     if run_dir.exists():
            #         shutil.rmtree(run_dir)
            #         self.logger.info(f"[process_molecule] Cleaned up run directory: {run_dir}")
            # except Exception as e:
            #     self.logger.warning(f"[process_molecule] Error during cleanup: {str(e)}")
            
            return {
                'type': 'success',
                'content': workflow_data,
                'metadata': {
                    'agent': 'ORCHESTRATOR',
                    'confidence': 1.0,
                    'reasoning': f'Successfully processed molecule through {workflow_type.value} workflow'
                }
            }
            
        except Exception as e:
            self.logger.error(f"[process_molecule] Error in process_molecule: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
                'metadata': {
                    'agent': 'ORCHESTRATOR',
                    'confidence': 0.0,
                    'reasoning': 'Failed to process molecule'
                }
            }

    def _validate_step_prerequisites(self, step: WorkflowStep, workflow_data: Dict) -> bool:
        """Validate that prerequisites are met for a given step."""
        # Empty requires list means no prerequisites
        if not step.requires:
            return True
            
        # Get current molecule data from context
        molecule_data = workflow_data.get('current_molecule', {})
            
        # Check each requirement group (OR conditions)
        for requirement_group in step.requires:
            # For each group, at least one requirement must be met
            requirements_met = False
            
            for req in requirement_group:
                # Check if step was completed in current workflow
                if workflow_data['completed_steps'].get(req, False):
                    requirements_met = True
                    break
                    
                # Check if data is available in molecule data
                if req == 'forward_prediction' and molecule_data.get('forward_predictions'):
                    requirements_met = True
                    break
                elif req == 'mol2mol' and molecule_data.get('mol2mol_results', {}).get('status') == 'success':
                    requirements_met = True
                    break
                elif req == 'nmr_simulation' and any(f'{type}_sim' in molecule_data.get('nmr_data', {}) 
                                                   for type in ['1H', '13C', 'HSQC', 'COSY']):
                    requirements_met = True
                    break
                elif req == 'peak_matching' and molecule_data.get('exp_sim_peak_matching', {}).get('status') == 'success':
                    requirements_met = True
                    break
                    
            if not requirements_met:
                self.logger.error(f"[process_molecule] Missing prerequisites for step '{step.keyword}'. Need one of: {requirement_group}")
                return False
                
        return True

    def _validate_step_output(self, step: WorkflowStep, result: Dict) -> bool:
        """Validate the output of a workflow step."""
        if result.get('type') != 'success':
            return False
            
        content = result.get('content', {})
        
        # Validate based on step keyword
        if step.keyword == 'error_thresholds':
            if not content or 'threshold_data' not in content:
                return False
            threshold_data = content['threshold_data']
            
            # Just check if status is success
            return threshold_data.get('status') == 'success'
        elif step.keyword == 'retrosynthesis':
            return 'predictions' in content
        elif step.keyword == 'nmr_simulation':
            # Check if NMR simulation data exists in master data format
            if not content or 'status' not in content:
                return False
            if content['status'] != 'success':
                return False
            if 'data' not in content:
                return False
            return True
        elif step.keyword == 'peak_matching':
            # Check for proper response format and success status
            if not content or 'exp_sim_peak_matching' not in content:
                return False
            peak_results = content['exp_sim_peak_matching']
            return peak_results.get('status') == 'success'
        elif step.keyword == 'mol2mol':
            # Check if mol2mol generation was successful
            if not content or 'status' not in content:
                return False
            return content['status'] == 'success'
        elif step.keyword == 'visual_comparison':
            return 'plots' in content
            
        return True  # No specific validation for other steps

    async def process(self, message: str, context: Dict = None) -> Dict:
        """Process an orchestration request."""
        try:
            self.logger.info(f"[process] Starting orchestration with context: {context}")
            self.logger.info(f"[process] Message: {message}")
            model_choice = context.get('model_choice', 'gemini-flash')
            processing_mode = context.get('processing_mode', 'batch')  # Default to batch
            
            # Load molecular data
            with open(self.molecular_data_file, 'r') as f:
                molecular_data = json.load(f)
            self.logger.info(f"[process] Loaded {len(molecular_data)} molecules")
            
            results = []
            total_molecules = len(molecular_data)
            successful = 0
            failed = 0

            # If we have a current molecule in context and single mode, process just that one
            if context and 'current_molecule' in context and processing_mode == 'single':
                current_molecule = context['current_molecule']
                sample_id = current_molecule.get('sample_id', 'unknown')
                self.logger.info(f"[process] Processing single molecule from context: {sample_id}")
                
                # Load full molecule data from JSON file if available
                if sample_id in molecular_data:
                    current_molecule = molecular_data[sample_id]  # Use complete data from JSON
                elif 'workflow_type' not in current_molecule:
                    # If molecule is not in JSON and doesn't have workflow type, determine it
                    workflow_type = determine_workflow_type(current_molecule)
                    current_molecule['workflow_type'] = workflow_type.value
                
                try:
                    # Use stored workflow type from molecule data
                    workflow_type = WorkflowType(current_molecule.get('workflow_type', WorkflowType.SPECTRAL_ONLY.value))
                    molecule_context = {
                        **(context or {}),
                        'workflow_type': workflow_type.value  # Add workflow type to context
                    }
                    result = await self.process_molecule(current_molecule, context=molecule_context)
                    results.append(result)
                    if result['type'] == 'success':
                        successful += 1
                    else:
                        failed += 1
                    self.logger.info(f"[process] Current molecule processed with status: {result['type']}")
                except Exception as e:
                    self.logger.error(f"[process] Error processing current molecule: {str(e)}")
                    failed += 1
                    results.append({
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'ORCHESTRATOR',
                            'confidence': 0.0,
                            'reasoning': 'Failed to process molecule'
                        }
                    })
                total_molecules = 1
            else:
                # Process all molecules from master JSON
                for molecule in molecular_data.values():
                    try:
                        self.logger.info(f"[process] Processing molecule {molecule.get('sample_id', 'unknown')}")
                        # Use stored workflow type from molecule data
                        workflow_type = WorkflowType(molecule.get('workflow_type', WorkflowType.SPECTRAL_ONLY.value))
                        molecule_context = {
                            **(context or {}),
                            'current_molecule': molecule,
                            'workflow_type': workflow_type.value  # Add workflow type to context
                        }
                        result = await self.process_molecule(molecule, context=molecule_context)
                        results.append(result)
                        
                        if result['type'] == 'success':
                            successful += 1
                        else:
                            failed += 1
                             
                        self.logger.info(f"[process] Molecule {molecule.get('sample_id', 'unknown')} processed with status: {result['type']}")
                    except Exception as e:
                        self.logger.error(f"[process] Error processing molecule {molecule.get('sample_id', 'unknown')}: {str(e)}")
                        failed += 1
                        results.append({
                            'type': 'error',
                            'content': str(e),
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Failed to process molecule'
                            }
                        })
                    
            response = {
                'total_molecules': total_molecules,
                'successful': successful,
                'failed': failed,
                'results': results
            }
            self.logger.info(f"[process] Orchestration complete. Success: {successful}, Failed: {failed}")
            return response

        except Exception as e:
            self.logger.error(f"[process] Fatal error in orchestration: {str(e)}")
            self.logger.error(f"[process] Traceback: {traceback.format_exc()}")
            raise
