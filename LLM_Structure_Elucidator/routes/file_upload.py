"""
File upload handling routes for the LLM Structure Elucidator.
"""
from flask import Blueprint, request, jsonify
import pandas as pd
from datetime import datetime
import json
import ast
import shutil
from rdkit import Chem
from rdkit.Chem import Descriptors
from models.molecule import MoleculeHandler
from utils.file_utils import save_uploaded_file, uploaded_smiles
from pathlib import Path
from typing import Dict, Any
from agents.orchestrator.workflow_definitions import determine_workflow_type, WorkflowType

file_upload = Blueprint('file_upload', __name__)

def parse_csv_data(row: pd.Series) -> Dict[str, Any]:
    """Parse all data from CSV row into a dictionary structure with molecule_data key.
    
    The structure will be:
    {
        'sample_id': str,
        'smiles': str,
        'molecule_data': {
            'smiles': str,
            'sample_id': str,
            'molecular_weight': float,
            'starting_materials': List[str],
            'timestamp': str,
            'nmr_data': {
                '1H_exp': {...},
                '13C_exp': {...},
                'HSQC_exp': {...},
                'COSY_exp': {...}
            }
        }
    }
    """
    # Get sample ID or generate one if not present
    sample_id = row.get('sample-id', f"SAMPLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    smiles = row.get('SMILES', '')
    
    molecule_data = {
        'smiles': smiles,
        'sample_id': sample_id,
        'starting_smiles': row.get('starting_smiles', '').split(';') if row.get('starting_smiles') else [],
        'timestamp': datetime.now().isoformat(),
        'nmr_data': {}
    }
    
    # Calculate molecular formula and weight using RDKit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecule_data['molecular_weight'] = Descriptors.ExactMolWt(mol)
            # Calculate molecular formula
            molecule_data['molecular_formula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
        else:
            print(f"Warning: Could not parse SMILES string for molecular calculations: {smiles}")
    except Exception as e:
        print(f"Error calculating molecular properties: {str(e)}")

    if '1H_NMR' in row and row['1H_NMR']:
        try:
            peaks = ast.literal_eval(row['1H_NMR'])
            molecule_data['nmr_data']['1H_exp'] = peaks
        except Exception as e:
            print(f"Error parsing 1H NMR data: {str(e)}")

    if '13C_NMR' in row and row['13C_NMR']:
        try:
            shifts = ast.literal_eval(row['13C_NMR'])
            molecule_data['nmr_data']['13C_exp'] = shifts
        except Exception as e:
            print(f"Error parsing 13C NMR data: {str(e)}")

    if 'HSQC' in row and row['HSQC']:
        try:
            correlations = ast.literal_eval(row['HSQC'])
            molecule_data['nmr_data']['HSQC_exp'] = correlations
        except Exception as e:
            print(f"Error parsing HSQC data: {str(e)}")

    if 'COSY' in row and row['COSY']:
        try:
            correlations = ast.literal_eval(row['COSY'])
            molecule_data['nmr_data']['COSY_exp'] = correlations
        except Exception as e:
            print(f"Error parsing COSY data: {str(e)}")

    return {
        'sample_id': sample_id,
        'smiles': row.get('SMILES', ''),
        'molecule_data': molecule_data
    }

def save_molecular_data(all_data: Dict[str, Any], filepath: str):
    """Save complete molecular data dictionary to a JSON file.
    Archives existing data file if present.
    
    Args:
        all_data: Dictionary containing all molecular data with sample_ids as keys
        filepath: Path to the source CSV file
    """
    data_dir = Path(__file__).parent.parent / "data" / "molecular_data"
    archive_dir = data_dir / "archive"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = data_dir / "molecular_data.json"
    
    # Archive existing file if it exists
    if json_file.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_file = archive_dir / f"molecular_data_{timestamp}.json"
        shutil.move(json_file, archive_file)
        print(f"Archived existing molecular_data.json to {archive_file}")
    
    # Save new data to file
    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=2)

def archive_molecular_data():
    """Archive existing molecular_data.json file if it exists."""
    molecular_data_dir = Path(__file__).parent.parent / "data" / "molecular_data"
    target_file = molecular_data_dir / "molecular_data.json"
    
    if target_file.exists():
        archive_dir = molecular_data_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"molecular_data_{timestamp}.json"
        shutil.move(target_file, archive_path)
        print(f"Archived existing molecular_data.json to {archive_path}")
        return True
    return False

@file_upload.route('/upload_file', methods=['POST'])
def upload_file():
    """Handle file upload for both CSV and JSON files."""
    try:
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        
        # Handle JSON file upload
        if file.filename.lower() == 'molecular_data.json':
            molecular_data_dir = Path(__file__).parent.parent / "data" / "molecular_data"
            target_file = molecular_data_dir / "molecular_data.json"
            
            # Archive existing file if present
            archive_molecular_data()
            
            # Save new file
            file.save(target_file)
            print(f"Saved new molecular_data.json to {target_file}")
            return jsonify({'message': 'Successfully uploaded molecular_data.json'})
            
        # Handle CSV file upload
        elif file.filename.endswith('.csv'):
            # Save the file
            filepath = save_uploaded_file(file)
            print(f"Saved file to: {filepath}")

            # Read CSV file
            df = pd.read_csv(filepath)
            print(f"Loaded CSV with columns: {df.columns.tolist()}")
            
            # Process each row and build complete data dictionary
            molecule_handler = MoleculeHandler()
            all_molecular_data = {}
            processed_samples = []
            
            for idx, row in df.iterrows():
                smiles = row['SMILES']
                if molecule_handler.validate_smiles(smiles):
                    # Parse data for this sample
                    data = parse_csv_data(row)

                    sample_id = data['sample_id']
                    
                    # Add source file and row index
                    data["molecule_data"]['source_file'] = str(filepath)
                    data["molecule_data"]['row_index'] = idx
                    
                    # Add workflow type
                    workflow_type = determine_workflow_type(data)
                    data["molecule_data"]['workflow_type'] = workflow_type.value
                    
                    # Add to complete dictionary
                    all_molecular_data[sample_id] = data#["molecule_data"]

                    processed_samples.append(sample_id)
            
            if not processed_samples:
                return jsonify({'error': 'No valid molecules found in CSV'}), 400
            
            # Save complete data dictionary
            save_molecular_data(all_molecular_data, filepath)
            
            return jsonify({
                'message': f'Successfully processed {len(processed_samples)} molecules',
                'sample_ids': processed_samples
            })
            
        else:
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Please upload a CSV file or molecular_data.json'}), 400
            
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_upload.route('/get_molecular_data', methods=['GET'])
def get_molecular_data():
    """Get all molecular data or filter by sample_id."""
    try:
        sample_id = request.args.get('sample_id')
        data_file = Path(__file__).parent.parent / "data" / "molecular_data" / "molecular_data.json"
        
        if not data_file.exists():
            return jsonify({'error': 'No molecular data found'}), 404
            
        with open(data_file, 'r') as f:
            all_data = json.load(f)
            
        if sample_id:
            if sample_id not in all_data:
                return jsonify({'error': f'Sample ID {sample_id} not found'}), 404
            return jsonify(all_data[sample_id])
        
        return jsonify(all_data)
            
    except Exception as e:
        print(f"Error getting molecular data: {str(e)}")
        return jsonify({'error': str(e)}), 500
