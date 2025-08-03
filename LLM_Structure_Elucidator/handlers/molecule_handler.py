"""
Molecule request handlers for Socket.IO events and HTTP requests.
"""
import os
import pandas as pd
import random
import json
from pathlib import Path
from flask import request, jsonify
from flask_socketio import emit
from core.socket import socketio
from utils.visualization import create_molecule_response
from rdkit import Chem
from core.app import app
from typing import Dict

# Path to molecular data JSON file
MOLECULAR_DATA_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data" / "molecular_data" / "molecular_data.json"

# Global variable to track current molecule
_current_molecule = None

def get_current_molecule():
    """Get the currently selected molecule."""
    global _current_molecule
    print(f"[Molecule Handler] Getting current molecule:")
    #print(f"  - Current molecule state: {json.dumps(_current_molecule, indent=2) if _current_molecule else None}")
    return _current_molecule

def set_current_molecule(smiles: str, name: str = None, nmr_data: Dict = None, sample_id: str = None):
    """Set the current molecule."""
    global _current_molecule
    print(f"\n[Molecule Handler] Setting current molecule:")
    print(f"  - SMILES: {smiles}")
    print(f"  - Name: {name}")
    print(f"  - Sample ID: {sample_id}")
    print(f"  - NMR Data Present: {bool(nmr_data)}")
    if nmr_data:
        print(f"  - NMR Data Keys: {list(nmr_data.keys())}")
        
        # Normalize NMR data keys to use _exp suffix
        normalized_nmr_data = {}
        key_mapping = {
            '1h': '1H_exp',
            '13c': '13C_exp',
            'hsqc': 'HSQC_exp',
            'cosy': 'COSY_exp',
            '1h_exp': '1H_exp',
            '13c_exp': '13C_exp',
            'hsqc_exp': 'HSQC_exp',
            'cosy_exp': 'COSY_exp',
            '1H': '1H_exp',
            '13C': '13C_exp',
            'HSQC': 'HSQC_exp',
            'COSY': 'COSY_exp'
        }
        
        for key, value in nmr_data.items():
            normalized_key = key_mapping.get(key.lower(), key)
            if value is not None:  # Only include non-null values
                normalized_nmr_data[normalized_key] = value
    else:
        normalized_nmr_data = {}
    
    _current_molecule = {
        'smiles': smiles,
        'name': name or 'Unknown',
        'sample_id': sample_id or 'unknown',
        'nmr_data': normalized_nmr_data
    }
    print(f"[Molecule Handler] Current molecule successfully set")
    #print(f"[Molecule Handler] Full molecule state: {json.dumps(_current_molecule, indent=2)}")

def get_molecular_data():
    """Get all molecular data from JSON storage."""
    print(f"\n[Molecule Handler] Attempting to read molecular data from: {MOLECULAR_DATA_PATH}")
    try:
        if not MOLECULAR_DATA_PATH.exists():
            print(f"[Molecule Handler] ERROR: Molecular data file not found at {MOLECULAR_DATA_PATH}")
            return None
            
        with open(MOLECULAR_DATA_PATH, 'r') as f:
            data = json.load(f)
            print(f"[Molecule Handler] Successfully loaded molecular data:")
            print(f"  - Number of molecules: {len(data)}")
            print(f"  - Sample IDs: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"[Molecule Handler] ERROR reading molecular data:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

def get_first_molecule_json():
    """Get the first molecule's data from JSON storage."""
    try:
        data = get_molecular_data()
        if not data:
            print("[Molecule Handler] Error: No molecular data found")
            return None
            
        # Get first molecule's data
        first_molecule_id = next(iter(data))
        first_molecule = data[first_molecule_id]
        
        # Set as current molecule
        set_current_molecule(
            smiles=first_molecule.get('smiles'),
            name=first_molecule.get('name', 'Unknown'),
            sample_id=first_molecule_id,
            nmr_data=first_molecule.get('nmr_data', {})
        )
        
        return {
            'status': 'success',
            'sample_id': first_molecule_id,
            'smiles': first_molecule.get('smiles'),
            # 'inchi': first_molecule.get('inchi'),
            # 'inchi_key': first_molecule.get('inchi_key'),
            'nmr_data': first_molecule.get('nmr_data', {})
        }
    except Exception as e:
        print(f"[Molecule Handler] Error getting first molecule: {str(e)}")
        return None

# Add Flask routes for JSON data access
@app.route('/get_molecular_data', methods=['GET'])
def handle_get_molecular_data():
    """Handle request to get all molecular data from JSON storage."""
    data = get_molecular_data()
    if data is None:
        return jsonify({"error": "No molecular data found"}), 404
    return jsonify({"status": "success", "data": data})

@app.route('/get_first_molecule_json', methods=['GET'])
def handle_get_first_molecule_json():
    """Handle request to get first molecule with all associated data from JSON."""
    result = get_first_molecule_json()
    if result is None:
        return jsonify({"error": "No molecules found in database"}), 404
    return jsonify(result)


def get_nmr_data_from_json(smiles: str):
    """Get NMR data for a SMILES string from JSON storage."""
    print(f"\n[Molecule Handler] Searching for NMR data for SMILES: {smiles}")
    try:
        data = get_molecular_data()
        if not data:
            print("[Molecule Handler] No molecular data available")
            return None
            
        # Find molecule with matching SMILES
        for sample_id, molecule in data.items():
            if molecule.get('smiles') == smiles:
                print(f"[Molecule Handler] Found NMR data for SMILES: {smiles} (sample-id: {sample_id})")
                
                # Get NMR data from the correct nested structure
                nmr_data = molecule.get('molecule_data', {}).get('nmr_data', {})
                result = {
                    'sample_id': sample_id,
                    '1H_exp': nmr_data.get('1H_exp'),
                    '13C_exp': nmr_data.get('13C_exp'),
                    'HSQC_exp': nmr_data.get('HSQC_exp'),
                    'COSY_exp': nmr_data.get('COSY_exp')
                }
                
                print(f"[Molecule Handler] Available spectra: {[k for k,v in result.items() if v and k != 'sample_id']}")
                return result
                
        print(f"[Molecule Handler] No NMR data found for SMILES: {smiles}")
        return {}
        
    except Exception as e:
        print(f"[Molecule Handler] ERROR retrieving NMR data:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

class MoleculeHandler:
    def generate_molecule_response(self, smiles):
        """Generate a complete molecule visualization response."""
        print(f"[Molecule Handler] Generating response for SMILES: {smiles}")
        try:
            response = create_molecule_response(smiles)
            print(f"[Molecule Handler] Response generated: {response is not None}")
            return response
        except Exception as e:
            print(f"[Molecule Handler] Error generating response: {str(e)}")
            print(f"[Molecule Handler] Error type: {type(e)}")
            return None

@socketio.on('get_molecule_image')
def get_molecule_image(data=None):
    """Generate and return a molecule image."""
    print("[Molecule Handler] Handling get_molecule_image request")
    try:
        # Get first molecule from JSON instead of random SMILES
        first_molecule = get_first_molecule_json()
        if not first_molecule or not first_molecule.get('smiles'):
            raise ValueError("Failed to get valid molecule from data")
            
        smiles = first_molecule['smiles']
        print(f"[Molecule Handler] Using SMILES: {smiles}")
        
        # Generate response
        response = create_molecule_response(smiles, is_3d=False)
        if response is None:
            raise ValueError("Failed to create molecule response")
        
        # Format molecular weight if present
        if 'molecular_weight' in response:
            response['molecular_weight'] = f"{response['molecular_weight']:.2f} g/mol"
        
        print("[Molecule Handler] Emitting molecule_image")
        emit('molecule_image', response)
        
    except Exception as e:
        error_msg = f"Failed to generate molecule image: {str(e)}"
        print(f"[Molecule Handler] Exception: {error_msg}")
        emit('message', {
            'content': error_msg,
            'type': 'error'
        })
