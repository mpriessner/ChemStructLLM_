"""
NMR Plot Agent for handling spectral visualization requests.
"""
from typing import Dict, Any, Optional, Tuple, List
from ..base.base_agent import BaseAgent
from services.llm_service import LLMService
import json
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NMRPlotAgent(BaseAgent):
    """Agent specialized in handling NMR plot requests and visualization."""
    
    def __init__(self, llm_service: LLMService):
        super().__init__(
            name="NMR Plot",
            capabilities=["plot", "show", "display", "visualize", "spectrum", "nmr", 
                         "hsqc", "proton", "carbon", "cosy", "1h", "13c"]
        )
        self.llm_service = llm_service
        self.plot_types = {
            "hsqc": ["hsqc", "heteronuclear single quantum coherence"],
            "proton": ["proton", "1h", "1h-nmr", "hydrogen"],
            "carbon": ["carbon", "13c", "13c-nmr"],
            "cosy": ["cosy", "correlation spectroscopy"]
        }
        
        # Default plot parameters
        self.default_parameters = {
            "hsqc": {
                "title": "HSQC NMR Spectrum",
                "x_label": "F2 (ppm)",
                "y_label": "F1 (ppm)",
                "style": "default"
            },
            "proton": {
                "title": "1H NMR Spectrum",
                "x_label": "Chemical Shift (ppm)",
                "y_label": "Intensity",
                "style": "default"
            },
            "carbon": {
                "title": "13C NMR Spectrum",
                "x_label": "Chemical Shift (ppm)",
                "y_label": "Intensity",
                "style": "default"
            },
            "cosy": {
                "title": "COSY NMR Spectrum",
                "x_label": "F2 (ppm)",
                "y_label": "F1 (ppm)",
                "style": "default"
            }
        }

    async def process(self, message: str,context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message to generate NMR visualizations.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        print("\n[NMR Plot Agent] ====== Starting Processing ======")
        print(f"[NMR Plot Agent] Message: {message}")
        model_choice = context.get('model_choice', 'gemini-flash')
        
        try:
            if not context or 'current_molecule' not in context:
                return {
                    "type": "error",
                    "content": "No molecule is currently selected. Please select a molecule first.",
                    "confidence": 0.0,
                    "reasoning": "Cannot generate NMR plot without a selected molecule"
                }
            
            logger.info(f"[NMR Plot Agent] Current molecule: {context['current_molecule']}")
            
            max_attempts = 3
            attempt = 1
            
            while attempt <= max_attempts:
                print(f"[NMR Plot Agent] Attempt {attempt} of {max_attempts}")
                
                # Use LLM for request analysis
                analysis_prompt = self._create_analysis_prompt(message, attempt > 1)
                analysis_response = await self.llm_service.get_completion(
                    message=analysis_prompt,
                    model=model_choice,
                    system="You are an NMR plot analysis assistant. Analyze plot requests and determine the appropriate visualization type and parameters. ONLY respond with the requested JSON format, no additional text."
                )
                
                plot_info = self._interpret_analysis(analysis_response)
                
                if plot_info.get("type") != "unknown":
                    if plot_info["confidence"] < 0.7:
                        return {
                            "type": "clarification",
                            "content": "I'm not quite sure which NMR plot you'd like to see. Could you specify if you want an HSQC, 1H (proton), 13C (carbon), or COSY spectrum?",
                            "confidence": plot_info["confidence"],
                            "reasoning": plot_info.get("reasoning", "Low confidence in plot type determination")
                        }
                    
                    # Create response
                    response = self._create_plot_response(plot_info["type"], plot_info["confidence"], plot_info.get("parameters"), context)
                    # Add reasoning to response
                    response["reasoning"] = plot_info.get("reasoning", "Successfully determined plot type and parameters")
                    print("[NMR Plot Agent] ====== Plot Request Processing Complete ======\n")
                    return response
                
                attempt += 1
            
            # If we've exhausted all attempts
            return {
                "type": "unknown",
                "confidence": 0.0,
                "content": "Unable to determine the appropriate NMR plot type after multiple attempts. Please rephrase your request.",
                "parameters": {},
                "reasoning": "Failed to determine plot type after multiple attempts"
            }
            
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error processing plot request: {str(e)}",
                "confidence": 0.0,
                "reasoning": f"An error occurred while processing the plot request: {str(e)}"
            }

    def _create_plot_response(self, plot_type: str, confidence: float, custom_params: Optional[Dict] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized plot response."""
        logger.info(f"Creating plot response for {plot_type} with confidence {confidence}")
        
        try:
            parameters = self.default_parameters.get(plot_type, {}).copy()

            # Get current SMILES from context if available
            current_smiles = None
            if context and 'current_molecule' in context:
                current_smiles = context['current_molecule'].get('SMILES')  # Note: Using uppercase SMILES
                print(f"[NMR Plot Agent] Current molecule SMILES from context: {current_smiles}")
            else:
                print("[NMR Plot Agent] No current molecule in context")
            
            # Generate NMR data using real data if available
            from utils.nmr_utils import generate_nmr_data
            nmr_data, is_random = generate_nmr_data(current_smiles, plot_type=plot_type, use_real_data=True)
            
            if not nmr_data:
                raise ValueError(f"No NMR data generated for {plot_type}")
            
            response = {
                "type": "plot",
                "plot_type": plot_type,
                "parameters": parameters,
                "confidence": confidence,
                "content": f"Displaying the {plot_type.upper()} spectrum for the current molecule."
            }
            
            if is_random:
                response["note"] = f"Note: Using simulated {plot_type.upper()} NMR spectrum as experimental data is not available."
            
            if plot_type in ['hsqc', 'cosy']:  # 2D NMR
                if len(nmr_data) == 3:
                    x_data, y_data, z_data = nmr_data
                    parameters['nmr_data'] = {
                        'x': x_data.tolist(),
                        'y': y_data.tolist(),
                        'z': z_data.tolist()
                    }
            else:  # 1D NMR
                x_data, y_data = nmr_data
                parameters['nmr_data'] = {
                    'x': x_data.tolist(),  # Convert numpy array to list
                    'y': y_data.tolist()   # Convert numpy array to list
                }

                response["content"] = f"Displaying the {plot_type.upper()} spectrum for the current molecule."
                # Only print non-data parameters to keep logs clean

            return response

        except Exception as e:
            logger.error(f"Error creating plot response: {str(e)}")
            return {
                "type": "error",
                "plot_type": plot_type,
                "parameters": self.default_parameters.get(plot_type, {}),
                "content": f"Error generating {plot_type.upper()} spectrum: {str(e)}",
                "confidence": 0.0,
                "reasoning": f"An error occurred while generating the plot response: {str(e)}"
            }
    
    def _create_analysis_prompt(self, message: str, is_retry: bool = False) -> str:
        """Create a prompt for analyzing the plot request."""
        retry_note = """
IMPORTANT: Previous attempt failed to generate valid JSON. 
Please ensure your response is ONLY valid JSON matching the required format below.
No additional text or explanations outside the JSON structure.""" if is_retry else ""

        return f"""Analyze this NMR plot request and determine the appropriate visualization.
Return ONLY a JSON response with NO additional text.{retry_note}

Rules for determining plot type:
1. If the request explicitly mentions '13C', 'carbon', or 'C13', use 'carbon' type
2. If the request explicitly mentions '1H', 'proton', or 'H1', use 'proton' type
3. If the request is ambiguous (just mentions 'NMR' or 'spectrum'), set confidence to 0.5
4. Default to 'carbon' type for requests about chemical shifts > 20 ppm
5. Default to 'proton' type for requests about chemical shifts < 10 ppm

Request: "{message}"

Available plot types:
- HSQC (Heteronuclear Single Quantum Coherence)
- 1H (Proton NMR)
- 13C (Carbon NMR)
- COSY (Correlation Spectroscopy)

Required JSON format:
{{
    "plot_request": {{
        "type": "plot_type",         # one of: hsqc, proton, carbon, cosy
        "confidence": 0.0,           # 0.0 to 1.0
        "reasoning": "explanation",   # Brief explanation of plot type choice. If confidence is low, explain why and suggest clearer prompts.
        "parameters": {{             # Optional parameters for the plot
            "title": "string",       # Custom title for the plot
            "x_label": "string",     # Custom x-axis label
            "y_label": "string",     # Custom y-axis label
            "style": "string"        # Plot style (default, publication, presentation)
        }}
    }}
}}"""
    
    def _interpret_analysis(self, analysis: str) -> Dict[str, Any]:
        """Interpret the LLM's analysis of the plot request."""
        try:
            # Clean and parse the JSON response
            content = analysis.get("content") if isinstance(analysis, dict) else analysis
            content = str(content).strip()
            
            # Clean markdown formatting if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json prefix
            content = content.replace("```", "").strip()
            
            # Parse JSON from content
            response = json.loads(content)

            # Extract plot request information
            plot_request = response.get("plot_request", {})
            plot_type = plot_request.get("type", "").lower()
            confidence = float(plot_request.get("confidence", 0.0))
            parameters = plot_request.get("parameters", {})
            reasoning = str(plot_request.get("reasoning", "No reasoning provided"))
            
            # Normalize plot type
            normalized_type = None
            for key, aliases in self.plot_types.items():
                if plot_type == key or plot_type in aliases:
                    normalized_type = key
                    break
            
            # Validate plot type
            if normalized_type is None:
                logger.info(f"[NMR Plot Agent] Unknown plot type: {plot_type}")
                return {
                    "type": "unknown",
                    "confidence": 0.0,
                    "parameters": {},
                    "reasoning": f"Could not determine NMR plot type from request: {plot_type}"
                }
            
            logger.info(f"[NMR Plot Agent] Normalized plot type '{plot_type}' to '{normalized_type}'")
            return {
                "type": normalized_type,
                "confidence": confidence,
                "parameters": parameters,
                "reasoning": reasoning
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"[NMR Plot Agent] Error interpreting analysis: {str(e)}")
            return {
                "type": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "reasoning": f"Error interpreting plot request: {str(e)}"
            }
    
    def get_available_plots(self) -> List[str]:
        """Return list of available plot types."""
        return list(self.plot_types.keys())
    
    def get_plot_description(self, plot_type: str) -> str:
        """Get a description of a specific plot type."""
        if plot_type not in self.plot_types:
            return "Unknown plot type"
            
        descriptions = {
            "hsqc": "2D NMR experiment showing correlations between directly bonded C-H pairs",
            "proton": "1D NMR spectrum showing hydrogen environments in the molecule",
            "carbon": "1D NMR spectrum showing carbon environments in the molecule",
            "cosy": "2D NMR experiment showing correlations between coupled protons"
        }
        return descriptions.get(plot_type, "No description available")
