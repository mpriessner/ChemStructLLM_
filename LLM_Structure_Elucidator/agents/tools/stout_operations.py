"""
Simple STOUT operations module for converting SMILES to IUPAC names.
"""
import uuid
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class STOUTOperations:
    """Simple handler for SMILES to IUPAC name conversion."""
    
    def __init__(self):
        """Initialize STOUT operations."""
        self.base_path = Path(__file__).parent.parent.parent
        self.temp_dir = self.base_path / "_temp_folder/stout"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = Path(__file__).parent.parent / "scripts" / "stout_local.sh"

    async def get_iupac_name(self, smiles: str) -> Dict[str, Any]:
        """Convert a single SMILES string to IUPAC name.
        
        Args:
            smiles: SMILES string to convert
            
        Returns:
            Dictionary with status and either IUPAC name or error message
        """
        job_id = uuid.uuid4().hex
        input_file = self.temp_dir / f"input_{job_id}.txt"
        output_file = self.temp_dir / f"output_{job_id}.json"
        
        try:
            # Write SMILES to input file
            input_file.write_text(smiles)
            
            # Run conversion script
            subprocess.run(
                [str(self.script_path), str(input_file), str(output_file), 'forward'],
                check=True,
                timeout=35
            )
            
            # Read and parse result
            result = json.loads(output_file.read_text())
            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'iupac_name': result['result']
                }
            else:
                return {
                    'status': 'error',
                    'error': result.get('error', 'Unknown conversion error')
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'error': 'Conversion timed out after 35 seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            # Cleanup temp files
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)