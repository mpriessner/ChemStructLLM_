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
            "Error handling",
            "Analysis coordination"
        ]
        super().__init__("Orchestration Agent", capabilities)
        self.llm_service = llm_service
        self.coordinator = coordinator
        self.tool_agent = coordinator.tool_agent if coordinator else None
        self.analysis_agent = coordinator.analysis_agent if coordinator else None
        self.logger = self._setup_logger()

        # Set path to molecular data file using relative path from orchestrator location
        self.molecular_data_file = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
        
        # Create log directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent / '_temp_folder' / 'memory' / 'logs'
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
                molecule_data = molecular_data[sample_id]
                # Set intermediate file path
                intermediate_path = self._get_intermediate_path(sample_id)
                molecule_data['_intermediate_file_path'] = str(intermediate_path)
                # Save initial data to intermediate file
                self._save_intermediate_data(sample_id, molecule_data)
                self.logger.info(f"[load_molecule_data] Successfully loaded fresh data for sample {sample_id}")
                return molecule_data
                    
            self.logger.error(f"[load_molecule_data] No data found for sample {sample_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"[load_molecule_data] Error loading molecule data: {str(e)}")
            return None

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get the path to the intermediate results file for a sample."""
        intermediate_dir = Path(__file__).parent.parent.parent / '_temp_folder' / 'intermediate_results'
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"[_get_intermediate_path] Using intermediate directory: {intermediate_dir}")
        return intermediate_dir / f"{sample_id}_intermediate.json"

    def _save_intermediate_data(self, sample_id: str, data: Dict) -> None:
        """Save data to the intermediate file."""
        filepath = self._get_intermediate_path(sample_id)
        self.logger.info(f"[_save_intermediate_data] Saving intermediate data to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_intermediate_data(self, sample_id: str) -> Dict:
        """Load data from the intermediate file."""
        filepath = self._get_intermediate_path(sample_id)
        if not filepath.exists():
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.logger.info(f"[_load_intermediate_data] Loaded intermediate data from {filepath}")
        return data
    
    def _update_master_data(self) -> None:
        """Update master molecular data file with workflow results."""
        try:
            with open(self.molecular_data_file, 'r') as f:
                master_data = json.load(f)
            
            # Get path to intermediate results directory using the same path as _get_intermediate_path
            intermediate_dir = Path(__file__).parent.parent.parent / '_temp_folder' / 'intermediate_results'
            self.logger.info(f"[_update_master_data] Checking intermediate directory: {intermediate_dir}")
            
            # Read all intermediate files and update master data
            updated_count = 0
            for intermediate_file in intermediate_dir.glob('*_intermediate.json'):
                try:
                    with open(intermediate_file, 'r') as f:
                        intermediate_data = json.load(f)
                    
                    # Extract sample_id from molecule_data
                    if 'molecule_data' in intermediate_data:
                        sample_id = intermediate_data['molecule_data'].get('sample_id')
                        if sample_id:
                            self.logger.info(f"[_update_master_data] Updating master data for sample {sample_id} from {intermediate_file.name}")
                            
                            # Create or update the sample entry in master data
                            if sample_id not in master_data:
                                master_data[sample_id] = {}
                            
                            # Update each top-level key from intermediate data
                            for key in intermediate_data.keys():
                                master_data[sample_id][key] = intermediate_data[key]
                            
                            updated_count += 1
                            self.logger.info(f"[_update_master_data] Updated keys for sample {sample_id}: {list(intermediate_data.keys())}")
                    else:
                        self.logger.warning(f"[_update_master_data] No molecule_data found in {intermediate_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"[_update_master_data] Error processing intermediate file {intermediate_file}: {str(e)}")
                    continue
            
            # Write updated master data back to file
            with open(self.molecular_data_file, 'w') as f:
                json.dump(master_data, f, indent=2)
            self.logger.info(f"[_update_master_data] Updated master molecular data file with {updated_count} molecules from intermediate files")
        
        except Exception as e:
            self.logger.error(f"[_update_master_data] Error updating master molecular data file: {str(e)}")
            

    async def process_molecule(self, molecule: Dict, user_input: str = None, context: Dict = None) -> Dict:
        """Process a single molecule through the structure elucidation workflow."""
        try:
            # Add detailed logging of input molecule structure
            self.logger.debug(f"[process_molecule] Input molecule structure: {json.dumps(molecule, indent=2)}")
            self.logger.info(f"[process_molecule] Molecule keys: {list(molecule.keys())}")
            
            if not self.tool_agent:
                raise RuntimeError("Tool agent not initialized")
            if not self.analysis_agent and 'analysis' in context.get('workflow_type', ''):
                raise RuntimeError("Analysis agent not initialized")

            # Use workflow type from context
            workflow_type = WorkflowType(context.get('workflow_type'))
            workflow_steps = get_workflow_steps(workflow_type)
            self.logger.info(f"[process_molecule] workflow_steps: {workflow_steps}")

            self.logger.info(f"[process_molecule] Using {len(workflow_steps)} workflow steps for workflow type: {workflow_type.value}")
            context = context or {}

            # Get sample_id
            sample_id = molecule.get('sample_id')
            self.logger.debug(f"[process_molecule] Attempting to get sample_id: {sample_id}")
            if not sample_id:
                # Try alternate location if sample_id not found at root
                sample_id = molecule.get('molecule_data', {}).get('sample_id')
                self.logger.debug(f"[process_molecule] Tried alternate location for sample_id: {sample_id}")
            if not sample_id:
                raise ValueError("No sample_id found in molecule or molecule_data")

            # Set intermediate file path
            intermediate_path = self._get_intermediate_path(sample_id)
            self.logger.debug(f"[process_molecule] Setting up intermediate path: {intermediate_path}")
            
            # Validate molecule_data exists
            if "molecule_data" not in molecule:
                self.logger.error(f"[process_molecule] molecule_data missing from input structure: {list(molecule.keys())}")
                raise KeyError("molecule_data not found in molecule structure")
            
            # Set intermediate file path
            intermediate_path = self._get_intermediate_path(sample_id)
            molecule["molecule_data"]['_intermediate_file_path'] = str(intermediate_path)
            
            # Save initial data to intermediate file
            workflow_progress = {
                'molecule_data': molecule["molecule_data"],  # Include the full molecule data
                'completed_steps': {},
                # 'step_outputs': {},
            }
            self._save_intermediate_data(sample_id, workflow_progress)
            
            # Execute each step in sequence
            for idx, step in enumerate(workflow_steps, 1):
                try:
                    self.logger.info(f"[process_molecule] Step {idx}/{len(workflow_steps)}: {step.description}")
                    self.logger.info(f"[process_molecule] Step keyword {idx}/{len(workflow_steps)}: {step.keyword}")
                    
                    # Load latest intermediate data
                    workflow_data = self._load_intermediate_data(sample_id)
                    self.logger.info(f"[process_molecule] Loaded workflow data keys: {list(workflow_data.keys())}")
                    
                    # Update context with current molecule data
                    context['current_molecule'] = workflow_data['molecule_data']
                    
                    # Execute the step command based on step type
                    if step.keyword == 'analysis':
                        # Use analysis agent for analysis steps
                        self.logger.info(f"[process_molecule] Executing analysis step: {step.command}")
                        self.logger.info(f"[process_molecule] Executing analysis keyword: {step.keyword}")

                        result = await self.analysis_agent.process_all({
                                'task_input': {
                                    'command': step.command,
                                    'workflow_data': workflow_data,
                                    # 'step_outputs': workflow_data['step_outputs']
                                },
                                'context': context
                            })
                    else:
                        # Use tool agent for other steps
                        result = await self.tool_agent.process(step.command, context=context)
                    
                    if result.get('type') == 'success':
                        workflow_data = self._load_intermediate_data(sample_id)
                        self.logger.info(f"[process_molecule] Step {idx} completed successfully")
                        workflow_data['completed_steps'][step.keyword] = True
                        # workflow_data['step_outputs'][step.keyword] = result.get('content', {})
                        self._save_intermediate_data(sample_id, workflow_data)

                    else:
                        error_msg = f"Step {idx} failed: {result.get('content', 'Unknown error')}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step execution failed'
                            }
                        }
                    
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
            # 
            # # Update master data file with results
            try:
                self._update_master_data()
                self.logger.info("[process_molecule] Successfully updated master data file")
            except Exception as e:
                self.logger.error(f"[process_molecule] Error updating master data file: {str(e)}")
                return {
                    'type': 'error',
                    'content': f"Workflow completed but failed to update master data: {str(e)}",
                    'metadata': {
                        'agent': 'ORCHESTRATOR',
                        'confidence': 0.0,
                        'reasoning': 'Failed to update master data file'
                    }
                }

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

    async def process(self, message: str, context: Dict = None) -> Dict:
        """Process an orchestration request."""
        try:
            self.logger.info(f"[process] Starting orchestration with context: {context}")
            self.logger.info(f"[process] Message: {message}")
            
            # Add logging for molecular data file loading
            self.logger.debug(f"[process] Loading molecular data from: {self.molecular_data_file}")
            try:
                with open(self.molecular_data_file, 'r') as f:
                    molecular_data = json.load(f)
                    self.logger.debug(f"[process] Molecular data file structure: {json.dumps({k: list(v.keys()) for k, v in molecular_data.items()}, indent=2)}")
            except Exception as e:
                self.logger.error(f"[process] Error loading molecular data file: {str(e)}")
                raise
            
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
                for current_molecule in molecular_data.values():
                    try:
                        self.logger.info(f"[process] Processing molecule {current_molecule.get('sample_id', 'unknown')}")
                        
                        # Determine or use stored workflow type
                        if 'workflow_type' not in current_molecule:
                            workflow_type = determine_workflow_type(current_molecule["molecule_data"])
                            self.logger.info(f"[process] Determined workflow type: {workflow_type.value}")
                            current_molecule["molecule_data"]['workflow_type'] = workflow_type.value
                        else:
                            workflow_type = WorkflowType(current_molecule['workflow_type'])
                            self.logger.info(f"[process] Using stored workflow type: {workflow_type.value}")
                        
                        molecule_context = {
                            **(context or {}),
                            'current_molecule': current_molecule["molecule_data"],
                            'workflow_type': workflow_type.value  # Add workflow type to context
                        }
                            
                        result = await self.process_molecule(current_molecule, context=molecule_context)
                        results.append(result)
                        
                        if result['type'] == 'success':
                            successful += 1
                        else:
                            failed += 1
                             
                        self.logger.info(f"[process] Molecule {current_molecule.get('sample_id', 'unknown')} processed with status: {result['type']}")
                    except Exception as e:
                        self.logger.error(f"[process] Error processing molecule {current_molecule.get('sample_id', 'unknown')}: {str(e)}")
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
            
            # Update master molecular data file with results
            self._update_master_data()
            
            self.logger.info(f"[process] Orchestration complete. Success: {successful}, Failed: {failed}")
            return response

        except Exception as e:
            self.logger.error(f"[process] Fatal error in orchestration: {str(e)}")
            self.logger.error(f"[process] Traceback: {traceback.format_exc()}")
            raise
