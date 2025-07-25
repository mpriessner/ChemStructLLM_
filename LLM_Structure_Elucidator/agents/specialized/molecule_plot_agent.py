"""
Specialized agent for handling molecule visualization requests.
"""
from typing import Dict, Any, Optional
from ..base.base_agent import BaseAgent
from services.llm_service import LLMService
from utils.visualization import create_molecule_response
from handlers.molecule_handler import get_molecular_data, get_nmr_data_from_json, set_current_molecule
import pandas as pd
import os
import random
import json

# Path to molecular data JSON file
TEST_SMILES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                               'data', 'molecular_data', 'molecular_data.json')

class MoleculePlotAgent(BaseAgent):
    """Agent for handling molecule visualization requests."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize the molecule plot agent."""
        super().__init__(
            name="Molecule Plot",
            capabilities=[
                "molecule visualization",
                "structure interpretation",
                "2D/3D rendering",
                "molecular property calculation"
            ]
        )
        self.llm_service = llm_service
        
    # def get_random_smiles(self) -> Optional[str]:
    #     """Get a random SMILES string from the molecular data JSON file."""
    #     try:
    #         # Get molecular data
    #         data = get_molecular_data()
    #         if not data:
    #             print("[MoleculePlotAgent] No molecular data found")
    #             return None
                
    #         # Get a random molecule
    #         sample_id = random.choice(list(data.keys()))
    #         molecule = data[sample_id]
    #         smiles = molecule.get('smiles')
            
    #         if not smiles:
    #             print(f"[MoleculePlotAgent] No SMILES found for sample {sample_id}")
    #             return None
            
    #         print(f"[MoleculePlotAgent] Selected random SMILES: {smiles}")
    #         return smiles
            
    #     except Exception as e:
    #         print(f"[MoleculePlotAgent] Error getting random SMILES: {str(e)}")
    #         return None

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message to generate molecule visualizations.
        
        Args:
            message: The user message to process
            model_choice: The LLM model to use (default: 'gemini-flash')
            context: Additional context for processing (optional)
        """
        print("\n[Molecule Plot Agent] ====== Starting Processing ======")
        print(f"[Molecule Plot Agent] Message: {message}")
        #print(f"[Molecule Plot Agent] Context: {json.dumps(context, indent=2) if context else None}")
        model_choice = context.get('model_choice', 'gemini-flash')
        
        try:
            # First, analyze the request using LLM
            print("[Molecule Plot Agent] Creating analysis prompt...")
            analysis_prompt = self._create_analysis_prompt(message)
            #print(f"[Molecule Plot Agent] Analysis Prompt:\n{analysis_prompt}")
            
            print("[Molecule Plot Agent] Getting analysis from LLM...")
            analysis_response = await self.llm_service.get_completion(
                message=analysis_prompt,
                model=model_choice,  # Using Gemini Flash for quick analysis
                system="You are a molecule visualization assistant. Extract SMILES strings and molecule indices from requests. ONLY respond with the requested JSON format, no additional text."
            )
            
            print("[Molecule Plot Agent] Raw LLM Response:")
            print(json.dumps(analysis_response, indent=2))
            
            # Interpret the analysis
            print("[Molecule Plot Agent] Interpreting analysis...")
            molecule_info = self._interpret_analysis(analysis_response)
            print(f"[Molecule Plot Agent] Interpreted info: {json.dumps(molecule_info, indent=2)}")
            
            if not molecule_info or not molecule_info.get("smiles"):
                print("[Molecule Plot Agent] No valid SMILES found")
                return {
                    "type": "tool_error",
                    "content": "Unable to determine molecule to visualize. Please provide a SMILES string or molecule index.",
                    "metadata": {
                        "reasoning": "Unable to determine molecule to visualize. Please provide a SMILES string or molecule index.",
                        "confidence": molecule_info.get("confidence", 0.0)
                    }
                }
            
            # Generate both 2D and 3D visualizations
            smiles = molecule_info["smiles"]
            print(f"[Molecule Plot Agent] Using SMILES: {smiles}")
            
            # Get NMR data for the molecule
            nmr_data = get_nmr_data_from_json(smiles)
            
            # Set as current molecule with NMR data
            sample_id = nmr_data.get('sample_id', 'unknown')
            set_current_molecule(
                smiles=smiles,
                nmr_data={
                    '1h': nmr_data.get('1h_exp'),
                    '13c': nmr_data.get('13c_exp'),
                    'hsqc': nmr_data.get('hsqc_exp'),
                    'cosy': nmr_data.get('cosy_exp')
                },
                sample_id=sample_id
            )
            
            response_2d = create_molecule_response(smiles, is_3d=False)
            response_3d = create_molecule_response(smiles, is_3d=True)
            
            if not response_2d or not response_3d:
                raise ValueError(f"Failed to create visualizations for SMILES: {smiles}")
            
            # Add NMR data to response
            response_2d['nmr_data'] = nmr_data
            response_3d['nmr_data'] = nmr_data
            
            # Format response
            response = {
                "type": "molecule_plot",
                "data": {
                    "2d": {
                        **response_2d,
                        "container": "vis-content-nmr-1"
                    },
                    "3d": {
                        **response_3d,
                        "container": "vis-content-nmr-3"
                    }
                },
                "molecule_index": molecule_info.get("molecule_index")
            }
            
            print("[Molecule Plot Agent] Response data:")
            # print(json.dumps(response, indent=2))
            print("[Molecule Plot Agent] Successfully generated visualization")
            print("[Molecule Plot Agent] ====== Processing Complete ======\n")
            return response
            
        except Exception as e:
            error_msg = f"Failed to process molecule visualization: {str(e)}"
            print(f"[Molecule Plot Agent] ERROR: {error_msg}")
            print(f"[Molecule Plot Agent] Error type: {type(e)}")
            import traceback
            print(f"[Molecule Plot Agent] Traceback:\n{traceback.format_exc()}")
            return {
                "type": "error",
                "content": error_msg
            }

    def _create_analysis_prompt(self, message: str) -> str:
        """Create the analysis prompt for the LLM."""
        return f"""Analyze this molecule visualization request and extract SMILES string and molecule index if present.
Return ONLY a JSON response with NO additional text.

Request: "{message}"

Required JSON format:
{{
    "molecule_request": {{
        "smiles": "string or null",      # SMILES string if explicitly mentioned in the request
        "molecule_index": "number or null", # Index of molecule if specified (e.g., "show molecule 2" -> 2)
        "confidence": 0.0,               # 0.0 to 1.0
        "reasoning": "explanation"        # Brief explanation of what was found in the request. If confidence is low, explain why and suggest alternative phrasing for clarity.
    }}
}}

Example responses:

1. "Show me the molecule with SMILES: CC(=O)O"
{{
    "molecule_request": {{
        "smiles": "CC(=O)O",
        "molecule_index": null,
        "confidence": 0.95,
        "reasoning": "Request contains explicit SMILES string CC(=O)O"
    }}
}}

2. "Display molecule 3"
{{
    "molecule_request": {{
        "smiles": null,
        "molecule_index": 3,
        "confidence": 0.9,
        "reasoning": "Request specifies molecule index 3"
    }}
}}

3. "Show me"
{{
    "molecule_request": {{
        "smiles": null,
        "molecule_index": null,
        "confidence": 0.2,
        "reasoning": "Request is too vague. No specific molecule or index specified. Suggest rephrasing to 'Show me molecule X' or 'Display SMILES: [specific SMILES string]' for clarity."
    }}
}}"""

    def _interpret_analysis(self, analysis: Any) -> Dict[str, Any]:
        """Interpret the LLM's analysis of the visualization request."""
        try:
            print(f"\n[Molecule Plot Agent] Starting analysis interpretation")
            print(f"[Molecule Plot Agent] Raw analysis input type: {type(analysis)}")
            # print(f"[Molecule Plot Agent] Raw analysis content: {analysis}")
            
            # Extract content from dict response
            content = analysis.get("content") if isinstance(analysis, dict) else analysis
            print(f"[Molecule Plot Agent] Extracted content: {content}")
            
            # Parse JSON from content
            response = json.loads(str(content).strip())
            print(f"[Molecule Plot Agent] Parsed JSON response: {json.dumps(response, indent=2)}")
            
            # Extract molecule request information
            molecule_request = response.get("molecule_request", {})
            smiles = molecule_request.get("smiles")
            molecule_index = molecule_request.get("molecule_index")
            confidence = float(molecule_request.get("confidence", 0.0))
            
            print(f"[Molecule Plot Agent] Extracted values:")
            print(f"  - SMILES: {smiles}")
            print(f"  - Molecule Index: {molecule_index}")
            print(f"  - Confidence: {confidence}")
            
            # If index is provided but no SMILES, get SMILES from index
            if molecule_index is not None and smiles is None:
                try:
                    # Get molecular data from JSON
                    data = get_molecular_data()
                    if not data:
                        print("[MoleculePlotAgent] No molecular data found")
                        return None

                    # Get sorted list of sample IDs to maintain consistent order
                    sample_ids = sorted(data.keys())
                    # Convert to 0-based index
                    index = molecule_index - 1 if molecule_index > 0 else 0
                    if 0 <= index < len(sample_ids):
                        sample_id = sample_ids[index]
                        molecule = data[sample_id]
                        smiles = molecule.get('smiles')
                        print(f"[MoleculePlotAgent] Using SMILES for sample {sample_id} at index {molecule_index}: {smiles}")
                        
                        # Get NMR data for indexed molecule
                        nmr_data = get_nmr_data_from_json(smiles)
                        if nmr_data:
                            # Set as current molecule with NMR data
                            set_current_molecule(
                                smiles=smiles,
                                nmr_data={
                                    'proton': nmr_data.get('proton'),
                                    'carbon': nmr_data.get('carbon'),
                                    'hsqc': nmr_data.get('hsqc'),
                                    'cosy': nmr_data.get('cosy')
                                },
                                sample_id=sample_id  # Use the actual sample ID from the data
                            )
                except Exception as e:
                    print(f"[MoleculePlotAgent] Error getting SMILES from index: {str(e)}")
            
            # If SMILES is directly provided (pasted into chat)
            elif smiles is not None:
                print(f"[MoleculePlotAgent] Using provided SMILES: {smiles}")
                # For pasted SMILES, set as current molecule but with no NMR data
                set_current_molecule(
                    smiles=smiles,
                    nmr_data=None,  # No NMR data for pasted SMILES
                    sample_id='unknown'
                )
            
            # If no index provided and no SMILES, use index 0
            else:
                try:
                    df = pd.read_csv(TEST_SMILES_PATH)
                    if len(df) > 0:
                        smiles = df.iloc[0]['SMILES']
                        molecule_index = 1  # 1-based index for user display
                        print(f"[MoleculePlotAgent] Using default SMILES at index 1: {smiles}")
                        
                        # Get NMR data for default molecule from JSON
                        nmr_data = get_nmr_data_from_json(smiles)
                        if nmr_data:
                            set_current_molecule(
                                smiles=smiles,
                                nmr_data={
                                    'proton': nmr_data.get('proton'),
                                    'carbon': nmr_data.get('carbon'),
                                    'hsqc': nmr_data.get('hsqc'),
                                    'cosy': nmr_data.get('cosy')
                                },
                                sample_id=nmr_data.get('sample_id', 'unknown')
                            )
                except Exception as e:
                    print(f"[MoleculePlotAgent] Error getting default SMILES: {str(e)}")
            
            result = {
                "smiles": smiles,
                "molecule_index": molecule_index,
                "confidence": confidence
            }
            print(f"[MoleculePlotAgent] Final interpreted result: {json.dumps(result, indent=2)}")

            return result
            
        except Exception as e:
            print(f"[MoleculePlotAgent] Error interpreting analysis: {str(e)}")
            print(f"[MoleculePlotAgent] Raw analysis input: {analysis}")
            return None