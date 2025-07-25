"""
Agent for managing and coordinating various tools in the system.
"""
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path
import uuid
from datetime import datetime
import pandas as pd
from ..base.base_agent import BaseAgent
from ..tools.nmr_simulation_tool import NMRSimulationTool
from ..tools.mol2mol_tool import Mol2MolTool
from ..tools.retro_synthesis_tool import RetrosynthesisTool
from ..tools.forward_synthesis_tool import ForwardSynthesisTool
from ..tools.peak_matching_tool import EnhancedPeakMatchingTool
# from ..tools.molecular_visual_comparison_tool import MolecularVisualComparisonTool
from ..tools.threshold_calculation_tool import ThresholdCalculationTool
from ..tools.candidate_analyzer_tool import CandidateAnalyzerTool
from ..tools.mmst_tool import MMSTTool
from ..tools.stout_tool import STOUTTool
from .config.tool_descriptions import TOOL_DESCRIPTIONS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolAgent(BaseAgent):
    """Agent responsible for managing and coordinating tool operations."""

    def __init__(self, llm_service):
        """Initialize the tool agent with available tools."""
        capabilities = [
            "NMR spectrum simulation",
            "Molecular analogue generation",
            "Retrosynthesis prediction",
            "Forward synthesis prediction",
            "Peak matching and comparison",
            "Threshold calculation",
            "Tool coordination",
            "Tool execution management",
            "SMILES/IUPAC name conversion"
        ]
        super().__init__("Tool Agent", capabilities)
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ToolAgent")
        
        # Initialize tools
        self.tools = {}
        self._initialize_tools()
        self.logger.info(f"Available tools after initialization: {list(self.tools.keys())}")

    def _initialize_tools(self):
        """Initialize and register available tools."""
        try:
            # Register standard tools
            self.logger.info("Registering standard tools...")
            self.tools['nmr_simulation'] = NMRSimulationTool()
            self.tools['mol2mol'] = Mol2MolTool()
            self.tools['retro_synthesis'] = RetrosynthesisTool()
            self.tools['forward_synthesis'] = ForwardSynthesisTool()
            self.tools['peak_matching'] = EnhancedPeakMatchingTool()
            # self.tools['molecular_visual_comparison'] = MolecularVisualComparisonTool()
            self.tools['threshold_calculation'] = ThresholdCalculationTool()
            self.tools['forward_candidate_analysis'] = CandidateAnalyzerTool(analysis_type='forward')
            self.tools['mol2mol_candidate_analysis'] = CandidateAnalyzerTool(analysis_type='mol2mol')
            self.tools['mmst_candidate_analysis'] = CandidateAnalyzerTool(analysis_type='mmst')
            self.tools['mmst'] = MMSTTool()
            self.tools['stout'] = STOUTTool()
            
            self.logger.info(f"Successfully registered all tools: {list(self.tools.keys())}")
        except Exception as e:
            self.logger.error(f"Error during tool initialization: {str(e)}", exc_info=True)
            raise

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message and route it to the appropriate tool.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        try:
            self.logger.info(f"Processing message: {message}")
            self.logger.debug(f"Available tools: {list(self.tools.keys())}")
            
            # Update context with processing mode
            context = context or {}
            
            # Get model choice from context, default to gemini-flash
            model_choice = context.get('model_choice', 'gemini-flash')
            tool_name = await self._determine_tool_llm(message, model_choice)
            self.logger.info(f"Selected tool: {tool_name}")
            # self.logger.info(f"context: {context}")

            if tool_name == 'nmr_simulation':
                # Get current molecule data and ensure sample_id
                molecule_data = context.get('current_molecule', {})
                sample_id = molecule_data.get('sample_id')
                context['use_slurm'] = False

                if not sample_id:
                    self.logger.error("No sample_id found in current molecule data")
                    return {'status': 'error', 'message': 'No sample_id found in current molecule data'}

                # Create context with just the necessary data
                nmr_context = {
                    'smiles': molecule_data.get('smiles'),
                    'sample_id': sample_id
                }
                
                self.logger.info(f"Running NMR simulation for sample {sample_id}")
                result = await self.tools['nmr_simulation'].simulate_nmr(sample_id, nmr_context)    
                
                return self._format_tool_response(result, "NMR simulation completed")

            elif tool_name == 'mol2mol':
                # Force local execution by setting use_slurm to False in context
                mol2mol_context = context.copy() if context else {}
                mol2mol_context['use_slurm'] = False

                if self._current_processing_type == 'batch':
                    self.logger.info("Processing all molecules for mol2mol generation")
                    result = await self.tools['mol2mol'].process_all_molecules()
                else:
                    # Single molecule mode requires current molecule
                    if not context or 'current_molecule' not in context:
                        return {
                            "type": "error",
                            "content": "No molecule data available. Please load or select a molecule first.",
                            "metadata": {
                                "agent": "TOOL_AGENT",
                                "confidence": 0.0,
                                "reasoning": "Missing required molecule data"
                            }
                        }
                    
                    self.logger.info("Processing single molecule for mol2mol generation")
                    molecule_data = context['current_molecule']
                    # self.logger.info(f"Molecule data received: {molecule_data}")
                    self.logger.info(f"Molecule data type: {type(molecule_data)}")
                    if isinstance(molecule_data, dict):
                        self.logger.info(f"Molecule data keys: {molecule_data.keys()}")
                        self.logger.info(f"SMILES present (uppercase): {'SMILES' in molecule_data}")
                        self.logger.info(f"SMILES present (lowercase): {'smiles' in molecule_data}")
                        smiles = molecule_data.get('SMILES') or molecule_data.get('smiles')
                        if smiles:
                            self.logger.info(f"SMILES value: {smiles}")
                            self.logger.info(f"Sample ID: {molecule_data.get('sample_id')}")
                    
                    # Check for either uppercase or lowercase SMILES
                    smiles = molecule_data.get('SMILES') or molecule_data.get('smiles') if isinstance(molecule_data, dict) else None
                    sample_id = molecule_data.get('sample_id') or molecule_data.get('sample-id') if isinstance(molecule_data, dict) else None
                    if smiles:
                        result = await self.tools['mol2mol'].generate_analogues(
                            smiles,
                            sample_id
                        )
                        self.logger.info(f"Mol2Mol result: {result}")
                        
                        # If successful, update master data and return simplified response
                        if result.get('status') == 'success':
                            # await self._update_master_data_with_mol2mol(result, molecule_data)
                            return {
                                'type': 'success',
                                'content': {'status': 'success'},  # Simplified response that satisfies orchestrator validation
                                'predictions': result.get('predictions', {})
                            }
                        else:
                            return {
                                'type': 'error',
                                'content': result.get('message', 'Unknown error in mol2mol generation'),
                                'metadata': {
                                    'agent': 'TOOL_AGENT',
                                    'confidence': 0.0,
                                    'reasoning': 'Mol2mol generation failed'
                                }
                            }
                    else:
                        return {
                            "type": "error",
                            "content": "Invalid molecule data format. Expected dictionary with SMILES key.",
                            "metadata": {
                                "agent": "TOOL_AGENT",
                                "confidence": 0.0,
                                "reasoning": "Invalid data format"
                            }
                        }
                
            elif tool_name == 'retro_synthesis':
                # Get current molecule data and ensure sample_id
                molecule_data = context.get('current_molecule') if context else None
                retro_context = context.copy() if context else {}
                retro_context['use_slurm'] = False
                
                if not molecule_data:
                    return {
                        "type": "error", 
                        "content": "No molecule data available. Please select a molecule first.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                    
                if isinstance(molecule_data, dict) and 'sample_id' not in molecule_data:
                    return {
                        "type": "error",
                        "content": "No sample ID found. Please ensure molecule has a sample ID.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required sample ID"
                        }
                    }
                molecule_data
                result = await self.tools['retro_synthesis'].predict_retrosynthesis(molecule_data, retro_context)
                return self._format_tool_response(result, "Retrosynthesis prediction completed")
            
            elif tool_name == 'forward_synthesis':
                # Get current molecule data and ensure sample_id
                molecule_data = context.get('current_molecule') if context else None
                forward_context = context.copy() if context else {}
                forward_context['use_slurm'] = False
                
                if not molecule_data:
                    return {
                        "type": "error", 
                        "content": "No molecule data available. Please select a molecule first.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                    
                if isinstance(molecule_data, dict) and 'sample_id' not in molecule_data:
                    return {
                        "type": "error",
                        "content": "No sample ID found. Please ensure molecule has a sample ID.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required sample ID"
                        }
                    }
                
                result = await self.tools['forward_synthesis'].predict_forward_synthesis(molecule_data, forward_context)
                return self._format_tool_response(result, "Forward synthesis prediction completed")
 
            elif tool_name == 'peak_matching':
                self.logger.info("=== Peak Matching Context ===")
                
                if not context or 'current_molecule' not in context:
                    return {
                        "type": "error",
                        "content": "No molecule data available for peak matching",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                
                molecule_data = context['current_molecule']
                sample_id = molecule_data.get('sample_id')
                if not sample_id:
                    return {
                        "type": "error",
                        "content": "No sample ID found in molecule data",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing sample ID"
                        }
                    }
                
                # Prepare context with comparison mode info
                peak_context = self._prepare_peak_matching_context(message, context)
                
                # Log the comparison mode
                comparison_mode = peak_context.get('comparison_mode', 'default')
                self.logger.info(f"Using comparison mode: {comparison_mode}")
                
                # Use new process_peaks method that handles intermediate files
                result = await self.tools['peak_matching'].process_peaks(sample_id, peak_context)
                
                # Format response based on comparison mode
                success_message = f"Peak matching completed using {comparison_mode} mode"
                return self._format_tool_response(result, success_message)
            
            # elif tool_name == 'molecular_visual_comparison':
            #     if not context:
            #         return {
            #             "type": "error",
            #             "content": "No context provided for molecular comparison.",
            #             "metadata": {
            #                 "agent": "TOOL_AGENT",
            #                 "confidence": 0.0,
            #                 "reasoning": "Missing required context"
            #             }
            #         }
                
            #     # Determine input type and prepare data based on LLM analysis
            #     input_data = {}
                
            #     # Batch processing with CSV
            #     if self._current_processing_type == 'batch':
            #         if 'guess_smiles_csv' not in context:
            #             return {
            #                 "type": "error",
            #                 "content": "CSV file with guess molecules not provided for batch processing.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required CSV file"
            #                 }
            #             }
                    
            #         if self._current_comparison_type == 'target' and 'target_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'batch_vs_target',
            #                 'guess_smiles_csv': context['guess_smiles_csv'],
            #                 'target_smiles': context['target_smiles']
            #             }
            #         elif self._current_comparison_type == 'starting_materials' and 'starting_materials_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'batch_vs_starting',
            #                 'guess_smiles_csv': context['guess_smiles_csv'],
            #                 'starting_materials_smiles': context['starting_materials_smiles']
            #             }
            #         else:
            #             return {
            #                 "type": "error",
            #                 "content": "Missing target or starting materials SMILES for batch comparison.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required SMILES"
            #                 }
            #             }
                
            #     # Single molecule comparison
            #     else:
            #         if 'guess_smiles' not in context:
            #             return {
            #                 "type": "error",
            #                 "content": "Guess molecule SMILES not provided.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required SMILES"
            #                 }
            #             }
                    
            #         if self._current_comparison_type == 'target' and 'target_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'guess_vs_target',
            #                 'guess_smiles': context['guess_smiles'],
            #                 'target_smiles': context['target_smiles']
            #             }
            #         elif self._current_comparison_type == 'starting_materials' and 'starting_materials_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'guess_vs_starting',
            #                 'guess_smiles': context['guess_smiles'],
            #                 'starting_materials_smiles': context['starting_materials_smiles']
            #             }
            #         else:
            #             return {
            #                 "type": "error",
            #                 "content": "Missing target or starting materials SMILES for comparison.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required SMILES"
            #                 }
            #             }
                
            #     # Create run directory with unique ID
            #     run_id = str(uuid.uuid4())
            #     run_dir = Path("_temp_folder") / "molecular_visual_comparison" / run_id
            #     run_dir.mkdir(parents=True, exist_ok=True)

            #     # Prepare context with comparison type
            #     comparison_context = {
            #         **context,
            #         'run_dir': str(run_dir),
            #         'run_id': run_id,
            #         'comparison_type': input_data['comparison_type']
            #     }

            #     self.logger.info(f"Running molecular visual comparison - {input_data['comparison_type']}")
            #     result = await self.tools['molecular_visual_comparison'].compare_structures(
            #         input_data=input_data,
            #         context=comparison_context
            #     )
            #     return self._format_tool_response(result, "Molecular visual comparison completed")

            elif tool_name == 'threshold_calculation':
                self.logger.info("[TOOL_AGENT] Initiating threshold calculation process")
                if not context or 'current_molecule' not in context:
                    return {
                        "type": "error",
                        "content": "Missing molecule data for threshold calculation",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Required molecule data context not provided"
                        }
                    }
                    
                try:
                    current_molecule = context['current_molecule']
                    threshold_tool = self.tools['threshold_calculation']
                    self.logger.info(f"[TOOL_AGENT] Calculating thresholds for molecule: {current_molecule.get('sample_id')}")
                    self.logger.info(f"[TOOL_AGENT] Calculating current_molecule: {current_molecule}")

                    sample_id = current_molecule.get('sample_id')

                    # Call calculate_threshold directly with lowercase smiles key
                    result = await threshold_tool.calculate_threshold(
                        sample_id=sample_id,  # Changed from SMILES to smiles
                        context=context
                    )
                    
                    # self.logger.info(f"[TOOL_AGENT] Threshold calculation completed: {result}")
                    return {
                        "type": "success",
                        "content": {'threshold_data': result},
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 1.0,
                            "reasoning": "Threshold calculation completed successfully"
                        }
                    }
                except Exception as e:
                    error_msg = f"[TOOL_AGENT] Threshold calculation failed: {str(e)}"
                    self.logger.error(error_msg)
                    return {
                        "type": "error",
                        "content": str(e),
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": error_msg
                        }
                    }
                
            elif 'candidate_analysis' in tool_name:  # Handle any type of candidate analysis
                self.logger.info(f"Starting {tool_name} processing")
                if not context or 'current_molecule' not in context:
                    self.logger.info(f"Context validation failed. Context exists: {bool(context)}, Keys in context: {context.keys() if context else 'None'}")
                    return {
                        "type": "error",
                        "content": "No molecule data available for candidate analysis",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                
                molecule_data = context['current_molecule']
                sample_id = molecule_data.get('sample_id')
                
                # Determine the analysis type from the tool name
                if tool_name == 'mol2mol_candidate_analysis':
                    analysis_type = 'mol2mol'
                elif tool_name == 'forward_candidate_analysis':
                    analysis_type = 'forward'
                elif tool_name == 'mmst_candidate_analysis':
                    analysis_type = 'mmst'
                else:
                    analysis_type = 'general'  # Fallback for generic candidate_analysis
                
                self.logger.info(f"Running candidate analysis with type: {analysis_type}")
                
                # Get the appropriate tool instance
                analyzer_tool = self.tools.get(tool_name)
                if not analyzer_tool:
                    self.logger.error(f"No tool found for {tool_name}")
                    return {
                        "type": "error",
                        "content": f"Tool not found: {tool_name}",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Tool initialization error"
                        }
                    }
                
                try:
                    result = await analyzer_tool.process(molecule_data, context)
                    return {
                        "type": "success",
                        "content": result,
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 1.0,
                            "reasoning": f"Successfully processed {analysis_type} candidate analysis"
                        }
                    }
                except Exception as e:
                    error_msg = f"Error in {analysis_type} candidate analysis: {str(e)}"
                    self.logger.error(error_msg)
                    return {
                        "type": "error",
                        "content": error_msg,
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Processing error"
                        }
                    }
                           
            elif tool_name == 'mmst':
                # Ensure we have the necessary context
                if not context or 'current_molecule' not in context:
                    return {
                        "type": "error",
                        "content": "No reference molecule data available. Please provide a reference molecule first.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }

                # Force local execution by default
                mmst_context = context.copy() if context else {}
                mmst_context['use_slurm'] = False

                # Get the reference molecule data
                molecule_data = context['current_molecule']
                smiles = molecule_data.get('SMILES') or molecule_data.get('smiles')
                molecule_id = molecule_data.get('sample_id') or molecule_data.get('ID')or molecule_data.get('sample-id')
                
                if not smiles:
                    return {
                        "type": "error",
                        "content": "Invalid molecule data format. Expected dictionary with SMILES key.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Invalid data format"
                        }
                    }

                try:
                    self.logger.info(f"Running MMST prediction for molecule {molecule_id} with SMILES: {smiles}")
                    mmst_result = await self.tools['mmst'].predict_structure(
                        reference_smiles=smiles,
                        molecule_id=molecule_id,
                        context=mmst_context
                    )
                    
                    if mmst_result['status'] == 'success':
                        # Update master data with MMST results
                        # await self._update_master_data_with_mmst(mmst_result, molecule_data)
                        
                        self.logger.info("MMST prediction completed successfully")
                        return {
                            'type': 'success',
                            'content': mmst_result,
                            'metadata': {
                                'agent': 'TOOL_AGENT',
                                'confidence': 1.0,
                                'reasoning': 'MMST prediction completed successfully'
                            }
                        }
                    else:
                        error_msg = mmst_result.get('message', 'Unknown error in MMST prediction')
                        self.logger.error(f"MMST prediction failed with error: {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'TOOL_AGENT',
                                'confidence': 0.0,
                                'reasoning': 'MMST prediction failed'
                            }
                        }
                except Exception as e:
                    self.logger.error(f"Exception during MMST prediction: {str(e)}", exc_info=True)
                    return {
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'TOOL_AGENT',
                            'confidence': 0.0,
                            'reasoning': 'Error during MMST prediction'
                        }
                    }

            # elif tool_name == 'stout':
            #     if not context:
            #         return {
            #             "type": "error",
            #             "content": "No context provided for STOUT conversion",
            #             "metadata": {
            #                 "agent": "TOOL_AGENT",
            #                 "confidence": 0.0,
            #                 "reasoning": "Missing required context"
            #             }
            #         }

            #     # Handle batch processing of molecules
            #     if 'molecules' in context:
            #         try:
            #             result = await self.tools['stout'].process_molecule_batch(context['molecules'])
            #             if result['status'] == 'success':
            #                 return {
            #                     'type': 'success',
            #                     'content': result,
            #                     'metadata': {
            #                         'agent': 'TOOL_AGENT',
            #                         'confidence': 1.0,
            #                         'reasoning': "Successfully processed molecule batch"
            #                     }
            #                 }
            #         except Exception as e:
            #             return {
            #                 'type': 'error',
            #                 'content': str(e),
            #                 'metadata': {
            #                     'agent': 'TOOL_AGENT',
            #                     'confidence': 0.0,
            #                     'reasoning': 'Error during batch processing'
            #                 }
            #             }

            #     # Handle single molecule conversion
            #     if 'input_str' not in context:
            #         return {
            #             "type": "error",
            #             "content": "No input string provided for conversion",
            #             "metadata": {
            #                 "agent": "TOOL_AGENT",
            #                 "confidence": 0.0,
            #                 "reasoning": "Missing required input"
            #             }
            #         }

            #     input_str = context['input_str']
            #     mode = context.get('conversion_mode', 'forward')  # Default to SMILES→IUPAC
                
            #     try:
            #         if mode == 'forward':
            #             result = await self.tools['stout'].convert_smiles_to_iupac(input_str)
            #         else:
            #             result = await self.tools['stout'].convert_iupac_to_smiles(input_str)
                        
            #         if result['status'] == 'success':
            #             return {
            #                 'type': 'success',
            #                 'content': result,
            #                 'metadata': {
            #                     'agent': 'TOOL_AGENT',
            #                     'confidence': 1.0,
            #                     'reasoning': f"Successfully converted {'SMILES to IUPAC' if mode == 'forward' else 'IUPAC to SMILES'}"
            #                 }
            #             }
            #         else:
            #             return {
            #                 'type': 'error',
            #                 'content': result['error'],
            #                 'metadata': {
            #                     'agent': 'TOOL_AGENT',
            #                     'confidence': 0.0,
            #                     'reasoning': 'Conversion failed'
            #                 }
            #             }
                        
            #     except Exception as e:
            #         return {
            #             'type': 'error',
            #             'content': str(e),
            #             'metadata': {
            #                 'agent': 'TOOL_AGENT',
            #                 'confidence': 0.0,
            #                 'reasoning': 'Error during conversion'
            #             }
            #         }

            return {
                "type": "error",
                "content": f"Unknown tool: {tool_name}",
                "metadata": {
                    "agent": "TOOL_AGENT",
                    "confidence": 0.0,
                    "reasoning": f"Tool '{tool_name}' not found in available tools"
                }
            }
                
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "metadata": {
                    "agent": "TOOL_AGENT",
                    "confidence": 0.0,
                    "reasoning": f"Tool execution failed: {str(e)}"
                }
            }
            
    def _format_tool_response(self, result: Dict, success_message: str) -> Dict:
        """Format tool response in a consistent way."""
        if isinstance(result, dict) and result.get('status') == 'error':
            return {
                "type": "error",
                "content": result.get('error', 'Unknown error'),
                "metadata": {
                    "agent": "TOOL_AGENT",
                    "confidence": 0.0,
                    "reasoning": result.get('error', 'Tool execution failed')
                }
            }
        
        # Create content dictionary with status field
        content = result if isinstance(result, dict) else {'data': result}
        if 'status' not in content:
            content['status'] = 'success'
        
        return {
            "type": "success",
            "content": content,
            "metadata": {
                "agent": "TOOL_AGENT",
                "confidence": 1.0,
                "reasoning": success_message
            }
        }

    # async def _update_master_data_with_mol2mol(self, mol2mol_result: Dict, molecule_data: Dict) -> None:
    #     """Update master data JSON with mol2mol results."""
    #     if mol2mol_result['status'] == 'success':
    #         # Get path to master data
    #         master_data_path = Path(__file__).parent.parent.parent / 'data' / 'molecular_data' / 'molecular_data.json'
            
    #         # Read the output file from mol2mol
    #         output_file = Path(mol2mol_result['output_file'])
    #         if output_file.exists():
    #             try:
    #                 # Read mol2mol results
    #                 mol2mol_df = pd.read_csv(output_file)
    #                 mol2mol_data = mol2mol_df.to_dict('records')
                    
    #                 # Extract target SMILES and suggestions
    #                 target_smiles = molecule_data.get('SMILES') or molecule_data.get('smiles')
    #                 suggestions = []
    #                 for entry in mol2mol_data:
    #                     # Each entry is a dictionary with one key-value pair
    #                     # The key is the target SMILES and value is the suggested SMILES
    #                     for _, suggestion in entry.items():
    #                         suggestions.append(suggestion)
                    
    #                 # Read existing master data
    #                 with open(master_data_path, 'r') as f:
    #                     master_data = json.load(f)
                    
    #                 # Get the sample_id
    #                 sample_id = molecule_data.get('sample_id') or molecule_data.get('sample-id')
    #                 if sample_id and sample_id in master_data:
    #                     # Add mol2mol results with new structure where target SMILES is the key
    #                     master_data[sample_id]['mol2mol_results'] = {
    #                         'generated_analogues_target': {
    #                             target_smiles: suggestions
    #                         },
    #                         'timestamp': datetime.now().isoformat(),
    #                         'status': 'success'
    #                     }
                        
    #                     # Write updated data back
    #                     with open(master_data_path, 'w') as f:
    #                         json.dump(master_data, f, indent=2)
                            
    #                     self.logger.info(f"Updated master data with mol2mol results for sample {sample_id}")
    #                 else:
    #                     self.logger.error(f"Sample ID {sample_id} not found in master data")
                        
    #             except Exception as e:
    #                 self.logger.error(f"Failed to update master data with mol2mol results: {str(e)}")

    
    def _analyze_peak_matching_request(self, message: str) -> Dict:
        """Analyze the peak matching request to determine comparison mode."""
        # Default to exp vs sim comparison
        comparison_mode = {
            'type': 'default',
            'input_data': {}
        }
        
        # Look for SMILES vs SMILES comparison
        if 'compare smiles' in message.lower():
            comparison_mode['type'] = 'smiles_vs_smiles'
            # Note: actual SMILES should be provided in context
            
        # Look for SMILES vs peaks comparison
        elif 'compare smiles with peaks' in message.lower():
            comparison_mode['type'] = 'smiles_vs_peaks'
            
        # Look for peaks vs CSV comparison
        elif 'compare peaks with csv' in message.lower():
            comparison_mode['type'] = 'peaks_vs_smiles_csv'
            
        return comparison_mode

    def _prepare_peak_matching_context(self, message: str, context: Dict) -> Dict:
        """Prepare context for peak matching based on request type."""
        # Analyze the request
        comparison_info = self._analyze_peak_matching_request(message)
        
        # Start with existing context or empty dict
        peak_context = context.copy() if context else {}
        
        # Add comparison mode info
        if 'input_data' not in peak_context:
            peak_context['input_data'] = {}
        peak_context['input_data'].update(comparison_info['input_data'])
        
        # Add comparison type
        peak_context['comparison_mode'] = comparison_info['type']
        
        return peak_context



    async def _update_master_data_with_mmst(self, mmst_result: Dict, molecule_data: Dict) -> None:
        """Update master data JSON with MMST results.
        Args:
            mmst_result: Results from MMST prediction containing processed predictions
            molecule_data: Original molecule data dictionary containing sample_id
        """
        if mmst_result['status'] == 'success':
            # Get path to master data
            master_data_path = Path(__file__).parent.parent.parent / 'data' / 'molecular_data' / 'molecular_data.json'
            
            try:
                # Extract predictions from MMST results
                predictions = mmst_result.get('predictions', {})
                
                # Read existing master data
                with open(master_data_path, 'r') as f:
                    master_data = json.load(f)
                
                # Get the sample_id
                sample_id = molecule_data.get('sample_id') or molecule_data.get('sample-id')
                if sample_id and sample_id in master_data:
                    # Add MMST results with structured data
                    master_data[sample_id]['mmst_results'] = {
                        'generated_molecules': predictions['generated_molecules'],
                        'model_info': predictions['model_info'],
                        'performance': predictions['performance'],
                        'timestamp': predictions['timestamp'],
                        'status': 'success'
                    }
                    
                    # Write updated data back
                    with open(master_data_path, 'w') as f:
                        json.dump(master_data, f, indent=2)
                        
                    self.logger.info(f"Updated master data with MMST results for sample {sample_id}")
                else:
                    self.logger.error(f"Sample ID {sample_id} not found in master data")
                    
            except Exception as e:
                self.logger.error(f"Failed to update master data with MMST results: {str(e)}")

#     def _prepare_peak_matching_input(self, context: Dict) -> Dict:
#         """Prepare input data for peak matching based on context.
        
#         Supports multiple comparison modes based on input_data:
#         1. smiles_vs_smiles: Compare two SMILES structures
#    2. peaks_vs_peaks: Compare two peak lists
#    3. smiles_vs_peaks: Compare SMILES against peak list
#    4. peaks_vs_csv: Compare peaks against SMILES CSV file
#    5. smiles_vs_csv: Compare reference SMILES against CSV file
#    6. exp_vs_sim: (Default) Compare experimental vs simulated peaks from master.json
#         """
#         # Check for explicit comparison modes in input_data
#         input_data = context.get('input_data', {})
#         # Log the full context
#         self.logger.debug(f"Full context: {context}")
        
#         # Extract input_data from context and log it
#         input_data = context.get('input_data', {})
#         self.logger.debug(f"Extracted input_data: {input_data}")
        
#         # SMILES vs SMILES comparison
#         if 'smiles1' in input_data and 'smiles2' in input_data:
#             self.logger.info("Using SMILES vs SMILES comparison mode")
#             return input_data
            
#         # Peaks vs Peaks comparison
#         if 'peaks1' in input_data and 'peaks2' in input_data:
#             self.logger.info("Using Peaks vs Peaks comparison mode")
#             return input_data
            
#         # SMILES vs Peaks comparison
#         if 'smiles' in input_data and 'peaks' in input_data:
#             self.logger.info("Using SMILES vs Peaks comparison mode")
#             return input_data
            
#         # Peaks vs SMILES CSV comparison
#         if 'peaks' in input_data and 'smiles_csv' in input_data:
#             self.logger.info("Using Peaks vs SMILES CSV comparison mode")
#             return input_data
            
#         # Reference SMILES vs CSV comparison
#         if 'reference_smiles' in input_data and 'smiles_csv' in input_data:
#             self.logger.info("Using Reference SMILES vs CSV comparison mode")
#             return input_data
            
#         # Default: Experimental vs Simulated peaks comparison
#         self.logger.info("Using default Experimental vs Simulated peaks comparison mode")
#         try:
#             master_path = Path('data/molecular_data/molecular_data.json')
#             if not master_path.exists():
#                 raise FileNotFoundError("master.json not found")
                
#             with open(master_path, 'r') as f:
#                 master_data = json.load(f)
                
#             sample_id = context['current_molecule']['sample_id']
#             if sample_id not in master_data:
#                 raise KeyError(f"Sample {sample_id} not found in master.json")
                
#             # Update molecule data from master.json
#             context['current_molecule'].update(master_data[sample_id])
#             nmr_data = context['current_molecule']['nmr_data']
            
#             # Verify we have both experimental and simulated data
#             required_exp = ['1H_exp', '13C_exp', 'HSQC_exp', 'COSY_exp']
#             required_sim = ['1H_sim', '13C_sim', 'HSQC_sim', 'COSY_sim']
            
#             if not all(key in nmr_data for key in required_exp + required_sim):
#                 missing = [key for key in required_exp + required_sim if key not in nmr_data]
#                 raise ValueError(f"Missing required NMR data: {missing}")
            
#             # Format peaks for 1D NMR (1H)
#             def format_1d_peaks(peaks):
#                 """Format 1D NMR peaks as parallel lists of shifts and intensities."""
#                 if not isinstance(peaks, list):
#                     return {'shifts': [], 'intensity': []}
#                 shifts = [shift for shift, _ in peaks]
#                 intensities = [1.0] * len(shifts)  # Constant intensity of 1.0
#                 return {
#                     'shifts': shifts,
#                     'intensity': intensities
#                 }

#             # Format peaks for 2D NMR (HSQC, COSY)
#             def format_2d_peaks(peaks):
#                 """Format 2D NMR peaks as parallel lists of F1 and F2 ppm values."""
#                 if not isinstance(peaks, list):
#                     return {'F2 (ppm)': [], 'F1 (ppm)': []}
#                 f2_values = [f2 for f2, _ in peaks]
#                 f1_values = [f1 for _, f1 in peaks]
#                 return {
#                     'F2 (ppm)': f2_values,
#                     'F1 (ppm)': f1_values
#                 }

#             # Format 13C peaks (only shifts, constant intensity)
#             def format_13c_peaks(peaks):
#                 """Format 13C NMR peaks with shifts and constant intensity."""
#                 if not isinstance(peaks, list):
#                     return {'shifts': [], 'intensity': []}
#                 return {
#                     'shifts': peaks,
#                     'intensity': [1.0] * len(peaks)
#                 }

#             return {
#                 'peaks1': {
#                     '1H': format_1d_peaks(nmr_data['1H_exp']),
#                     '13C': format_13c_peaks(nmr_data['13C_exp']),
#                     'HSQC': format_2d_peaks(nmr_data['HSQC_exp']),
#                     'COSY': format_2d_peaks(nmr_data['COSY_exp'])
#                 },
#                 'peaks2': {
#                     '1H': format_1d_peaks(nmr_data['1H_sim']),
#                     '13C': format_13c_peaks(nmr_data['13C_sim']),
#                     'HSQC': format_2d_peaks(nmr_data['HSQC_sim']),
#                     'COSY': format_2d_peaks(nmr_data['COSY_sim'])
#                 },
#                 'matching_mode': 'hung_dist_nn',
#                 'error_type': 'sum',
#                 'spectra': ['1H', '13C', 'HSQC', 'COSY']
#             }
            
#         except Exception as e:
#             self.logger.error(f"Error preparing experimental vs simulated comparison: {str(e)}")
#             return None

    async def _determine_tool_llm(self, message: str, model_choice: str) -> str:
        """Use LLM to determine which tool to use based on the message content."""
        self.logger.info(f"Determining tool for message: {message}")
        self.logger.debug(f"Available tools: {list(TOOL_DESCRIPTIONS.keys())}")
        
        system_prompt = f"""You are a tool selection agent for a molecular analysis system. Your task is to analyze the user's message and select the most appropriate tool based on their needs.

Available tools and their capabilities:

{json.dumps(TOOL_DESCRIPTIONS, indent=2)}

Pay special attention to:
1. Processing scope:
       - Single molecule: When the user wants to process a specific molecule
   - Batch processing: When the task requires processing multiple molecules, such as:
     * Retrosynthesis predictions (always batch)
     * Forward synthesis predictions (always batch)
     * When explicitly requested ("all molecules", "every compound")
     * When comparing against a database or set of molecules

2. Type of processing:
   - Direct operations (e.g., NMR simulation on one molecule)
   - Comparative operations (e.g., peak matching between molecules)
   - Predictive operations (e.g., retrosynthesis, which needs context from all molecules)

3. Input sources:
   - Individual SMILES strings
   - CSV files
   - Molecular databases
   - Experimental data

Analyze the user's message and select the most appropriate tool. Return your response in JSON format with the following structure:
{{
    "selected_tool": "tool_name",
    "confidence": 0.0,  # 0.0 to 1.0
    "reasoning": "explanation of why this tool was selected",
    "processing_type": "single or batch",  # Indicate if batch processing is needed
    "comparison_type": "target or starting_materials",  # Type of comparison needed (if applicable)
    "batch_reason": "explanation of why batch processing was selected" # Only if processing_type is "batch"
}}

Please respond with the selected tool name."""

        try:
            self.logger.info("Sending request to LLM service")
            response = await self.llm_service.get_completion(
                message=message,
                system=system_prompt,
                require_json=True,
                model=model_choice
            )
            
            self.logger.debug(f"LLM response: {response}")
            result = json.loads(response)
            self.logger.info(f"Selected tool: {result['selected_tool']} with confidence {result['confidence']}")
            self.logger.debug(f"Selection reasoning: {result['reasoning']}")
            
            if result['confidence'] < 0.7:  # Confidence threshold
                self.logger.warning(f"Low confidence in tool selection: {result['confidence']}")
                raise ValueError(f"Low confidence in tool selection: {result['confidence']}")
            
            # Store additional context for tool use
            self._current_processing_type = result.get('processing_type', 'single')
            self._current_comparison_type = result.get('comparison_type', None)
                
            return result['selected_tool']
            
        except Exception as e:
            self.logger.error(f"Error determining tool: {str(e)}", exc_info=True)
            raise ValueError(f"Error determining tool: {str(e)}")

    def supports_message(self, message: str) -> bool:
        """Check if this agent can handle the given message."""
        # Always return True since we let the LLM decide
        return True
