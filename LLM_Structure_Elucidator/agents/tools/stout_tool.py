"""
STOUT (Structure to IUPAC Name) tool for SMILES/IUPAC name conversion.
"""

import json
import uuid
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional, List, Union
from .data_extraction_tool import DataExtractionTool

class STOUTTool:
    """Tool for converting between SMILES and IUPAC names using STOUT."""
    
    def __init__(self):
        """Initialize the STOUT tool."""
        # Get path relative to this file's location
        self.base_path = Path(__file__).parent.parent.parent
        self.stout_dir = self.base_path / "_temp_folder/stout"
        self.stout_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = Path(__file__).parent.parent / "scripts" / "stout_local.sh"
        self.data_tool = DataExtractionTool()
        self.intermediate_dir = self.base_path / "_temp_folder" / "intermediate_results"

    
    async def convert_smiles_to_iupac(self, smiles: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert SMILES to IUPAC name.
        
        Args:
            smiles: The SMILES string to convert
            context: Optional context dictionary
            
        Returns:
            Dictionary containing conversion result or error
        """
        result = await self._run_conversion(smiles, "forward")
        return result
    
    async def convert_iupac_to_smiles(self, iupac: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert IUPAC name to SMILES."""
        return await self._run_conversion(iupac, "reverse")
    
    async def process_molecule_batch(self, molecules: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a batch of molecules to get their IUPAC names.
        
        Args:
            molecules: List of dictionaries containing molecule information with SMILES
            context: Optional context dictionary
            
        Returns:
            Dictionary containing batch conversion results
        """
        # Extract sample_id from context
        sample_id = None
        if context and 'sample_id' in context:
            sample_id = context['sample_id']
        elif molecules and isinstance(molecules[0], dict) and 'sample_id' in molecules[0]:
            sample_id = molecules[0]['sample_id']
            
        if not sample_id:
            raise ValueError("No sample_id provided in context or molecules")

        # Load or create intermediate data
        intermediate_data = self._load_or_create_intermediate(sample_id, context)
        
        # Generate unique filenames for batch processing
        job_id = uuid.uuid4().hex
        input_file = self.stout_dir / f"batch_input_{job_id}.json"
        output_file = self.stout_dir / f"batch_output_{job_id}.json"
        
        try:
            # Prepare batch input in required format
            batch_input = []
            for molecule in molecules:
                smiles = molecule.get('SMILES') or molecule.get('smiles')
                if smiles:
                    batch_input.append({
                        'smiles': smiles,
                        'molecule_id': molecule.get('sample_id'),
                        'original_data': molecule
                    })
            
            if not batch_input:
                return {
                    'status': 'error',
                    'error': 'No valid SMILES found in input molecules'
                }
            
            # Write batch input to file
            input_file.write_text(json.dumps(batch_input))
            
            # Run conversion script with batch mode
            subprocess.run(
                [str(self.script_path), str(input_file), str(output_file), 'forward', '--batch'],
                check=True,
                timeout=30  # Align timeout with shell script
            )
            
            # Read and parse results
            results = json.loads(output_file.read_text())
            
            # Update original molecules with IUPAC names and save to intermediate file
            for result, orig_molecule in zip(results, molecules):
                if result['status'] == 'success':
                    orig_molecule['iupac_name'] = result['result']
                    # intermediate_data['step_outputs']['stout'][orig_molecule.get('sample_id')] = {
                    #     'smiles': orig_molecule.get('smiles'),
                    #     'iupac_name': result['result'],
                    #     'status': 'success'
                    # }
                else:
                    orig_molecule['iupac_error'] = result.get('error', 'Unknown conversion error')
                    # intermediate_data['step_outputs']['stout'][orig_molecule.get('sample_id')] = {
                    #     'smiles': orig_molecule.get('smiles'),
                    #     'iupac_name': None,
                    #     'status': 'error'
                    # }
            
            # Save updated intermediate data
            self._save_intermediate(sample_id, intermediate_data)
            
            return {
                'status': 'success',
                'results': results,
                'updated_molecules': molecules  # Return updated molecules for caller's use
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'error': 'Batch conversion timed out after 30 seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Batch processing failed: {str(e)}'
            }
        finally:
            # Cleanup
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

    async def _run_conversion(self, input_str: str, mode: str) -> Dict[str, Any]:
        """Run the conversion process."""
        # Generate unique filenames
        job_id = uuid.uuid4().hex
        input_file = self.stout_dir / f"input_{job_id}.txt"
        output_file = self.stout_dir / f"output_{job_id}.json"
        
        try:
            # Write input to file
            input_file.write_text(input_str)
            
            # Run conversion script
            subprocess.run(
                [str(self.script_path), str(input_file), str(output_file), mode],
                check=True,
                timeout=35
            )
            
            # Read and parse result
            result = json.loads(output_file.read_text())
            return result
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Conversion timed out",
                "mode": mode
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "mode": mode
            }
        finally:
            # Cleanup
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

    async def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data."""
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                # if 'step_outputs' not in data:
                #     data['step_outputs'] = {}
                # if 'stout' not in data['step_outputs']:
                #     data['step_outputs']['stout'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = self.base_path / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)
                if sample_id in master_data:
                    # Create new intermediate with this sample's data
                    intermediate_data = {
                        'molecule_data': master_data[sample_id],
                        # 'step_outputs': {'stout': {}}
                    }
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
        
        # Create empty intermediate if no data found
        intermediate_data = {
            'molecule_data': {},
            # 'step_outputs': {'stout': {}}
        }
        self._save_intermediate(sample_id, intermediate_data)
        return intermediate_data

    def _save_intermediate(self, sample_id: str, data: Dict) -> None:
        """Save data to intermediate file."""
        intermediate_path = self._get_intermediate_path(sample_id)
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"
