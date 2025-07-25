"""
Agent for modifying script parameters in bash files.
"""
from typing import Dict, Any, Optional, List, Union
import re
from pathlib import Path
import logging
from ..base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ScriptModifierAgent(BaseAgent):
    """Agent for modifying parameters in bash scripts while preserving structure."""
    
    def __init__(self):
        capabilities = [
            "Modify bash script parameters",
            "Preserve script structure",
            "Handle multiple parameter updates",
            "Support local and SLURM scripts"
        ]
        super().__init__("Script Modifier", capabilities)
        
        # Define script-specific parameter configurations
        self.script_configs = {
            "chemformer_forward": {
                "pattern": r".*chemformer_forward.*\.sh$",
                "parameters": {
                    "BATCH_SIZE": {"type": int, "min": 1, "description": "Batch size for predictions"},
                    "N_BEAMS": {"type": int, "min": 1, "description": "Number of beams for beam search"},
                    "N_UNIQUE_BEAMS": {"type": (int, str), "allowed_str": ["None"], "min": 1, "description": "Number of unique beams"}
                }
            },
            "chemformer_retro": {
                "pattern": r".*chemformer_retro.*\.sh$",
                "parameters": {
                    "BATCH_SIZE": {"type": int, "min": 1, "description": "Batch size for predictions"},
                    "N_BEAMS": {"type": int, "min": 1, "description": "Number of beams for beam search"},
                    "N_UNIQUE_BEAMS": {"type": (int, str), "allowed_str": ["None"], "min": 1, "description": "Number of unique beams"}
                }
            },
            "mol2mol": {
                "pattern": r".*mol2mol.*\.sh$",
                "parameters": {
                    "DELTA_WEIGHT": {"type": int, "min": 1, "description": "Delta weight parameter"},
                    "TANIMOTO_FILTER": {"type": float, "min": 0.0, "max": 1.0, "description": "Tanimoto filter threshold"},
                    "NUM_GENERATIONS": {"type": int, "min": 1, "description": "Number of generations"},
                    "MAX_TRIALS": {"type": int, "min": 1, "description": "Maximum number of trials"},
                    "MAX_SCAFFOLD_GENERATIONS": {"type": int, "min": 1, "description": "Maximum scaffold generations"}
                }
            },
            "sgnn": {
                "pattern": r".*sgnn.*\.sh$",
                "parameters": {}  # SGNN script uses command line arguments directly
            }
        }

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message to modify scripts.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        print("\n[Script Modifier Agent] ====== Starting Processing ======")
        print(f"[Script Modifier Agent] Message: {message}")
        model_choice = context.get('model_choice', 'gemini-flash')
        processing_mode = context.get('processing_mode', 'single')
        try:
            if isinstance(message, str):
                message = [message]
                
            results = {}
            for script_path in message:
                path = Path(script_path)
                if not path.exists():
                    results[script_path] = {
                        'status': 'error',
                        'message': f'Script not found: {script_path}'
                    }
                    continue
                
                try:
                    # Identify script type and get its config
                    script_config = self._get_script_config(path.name)
                    if not script_config:
                        results[script_path] = {
                            'status': 'error',
                            'message': f'Unsupported script type: {path.name}'
                        }
                        continue
                    
                    # Validate parameters against script-specific rules
                    invalid_params = self._validate_parameters({}, script_config["parameters"])
                    if invalid_params:
                        results[script_path] = {
                            'status': 'error',
                            'message': f'Invalid parameters: {", ".join(invalid_params)}'
                        }
                        continue
                    
                    # Read current script content
                    with open(path, 'r') as f:
                        content = f.read()
                    
                    # Modify parameters
                    modified_content = self._modify_parameters(content, {})
                    
                    # Write modified content back
                    with open(path, 'w') as f:
                        f.write(modified_content)
                    
                    results[script_path] = {
                        'status': 'success',
                        'message': f'Successfully modified parameters in {path.name}',
                        'modified_parameters': []
                    }
                    
                except Exception as e:
                    results[script_path] = {
                        'status': 'error',
                        'message': f'Error modifying {path.name}: {str(e)}'
                    }
            
            return {
                'status': 'success' if all(r['status'] == 'success' for r in results.values()) else 'partial_success',
                'results': results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error in script modification: {str(e)}'
            }

    def _get_script_config(self, script_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific script type."""
        for config in self.script_configs.values():
            if re.match(config["pattern"], script_name):
                return config
        return None

    def _validate_parameters(self, parameters: Dict[str, Any], script_params: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate parameters against script-specific rules."""
        invalid_params = []
        for param_name, value in parameters.items():
            if param_name not in script_params:
                invalid_params.append(f'{param_name} (unknown parameter)')
                continue
                
            param_config = script_params[param_name]
            
            # Type validation
            if not isinstance(value, param_config["type"]):
                if isinstance(param_config["type"], tuple):
                    # Handle special cases like N_UNIQUE_BEAMS that can be int or "None"
                    if isinstance(value, str) and value in param_config.get("allowed_str", []):
                        continue
                invalid_params.append(f'{param_name} (invalid type: expected {param_config["type"]}, got {type(value)})')
                continue
            
            # Range validation for numeric types
            if isinstance(value, (int, float)):
                if "min" in param_config and value < param_config["min"]:
                    invalid_params.append(f'{param_name} (value below minimum: {param_config["min"]})')
                if "max" in param_config and value > param_config["max"]:
                    invalid_params.append(f'{param_name} (value above maximum: {param_config["max"]})')
                    
        return invalid_params

    def _modify_parameters(self, content: str, parameters: Dict[str, Any]) -> str:
        """Modify parameters in script content while preserving structure."""
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            # Check if line defines a parameter we want to modify
            for param_name, new_value in parameters.items():
                # Match parameter definition (handles different formats)
                pattern = rf'^{param_name}=.*$'
                if re.match(pattern, line.strip()):
                    # Preserve any comments that might be on the same line
                    comment = ''
                    if '#' in line:
                        comment = line[line.index('#'):]
                    
                    # Format the new value appropriately
                    if isinstance(new_value, str) and not new_value.startswith('"'):
                        formatted_value = f'"{new_value}"'
                    else:
                        formatted_value = str(new_value)
                    
                    # Create new line with preserved spacing
                    leading_space = len(line) - len(line.lstrip())
                    line = ' ' * leading_space + f'{param_name}={formatted_value}{comment}'
                    break
            
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)

    def supports_message(self, message: str) -> bool:
        """Check if this agent can handle the given message."""
        return any(keyword in message.lower() for keyword in [
            'modify script', 'update parameter', 'change parameter',
            'script parameter', 'bash script', 'shell script'
        ])
