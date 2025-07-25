from flask import jsonify
from pathlib import Path
import json

def init_data_routes(app):
    @app.route('/get_molecular_data', methods=['GET'])
    def get_molecular_data():
        try:
            json_path = Path(app.root_path) / "data" / "molecular_data" / "molecular_data.json"
            if not json_path.exists():
                return jsonify({"error": "No molecular data found"}), 404
                
            with open(json_path, 'r') as f:
                data = json.load(f)
            return jsonify({"status": "success", "data": data})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/get_first_molecule_json', methods=['GET'])
    def get_first_molecule_json():
        try:
            json_path = Path(app.root_path) / "data" / "molecular_data" / "molecular_data.json"
            if not json_path.exists():
                return jsonify({"error": "No molecular data found"}), 404
                
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if not data:
                return jsonify({"error": "No molecules found in database"}), 404
                
            # Get the first molecule's data
            first_molecule_id = next(iter(data))
            first_molecule = data[first_molecule_id]
            
            return jsonify({
                "status": "success",
                "sample_id": first_molecule_id,
                "smiles": first_molecule.get("smiles"),
                "inchi": first_molecule.get("inchi"),
                "inchi_key": first_molecule.get("inchi_key"),
                "nmr_data": first_molecule.get("nmr_data", {})
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
