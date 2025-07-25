"""
Visualization utilities for molecule and plot generation.
"""
import io
import base64
import numpy as np
import plotly.graph_objects as go
from handlers.molecule_handler import MoleculeHandler
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from PIL import Image

def create_molecule_image(mol, size=(400, 400)):
    """Create a 2D PIL Image of the molecule."""
    print(f"\n[Visualization] Creating 2D molecule image:")
    print(f"  - Molecule type: {type(mol)}")
    print(f"  - Molecule SMILES: {Chem.MolToSmiles(mol) if mol else None}")
    print(f"  - Requested size: {size}")
    
    try:
        # Generate 2D depiction
        img = Draw.MolToImage(mol, size=size)
        print("[Visualization] Successfully generated 2D molecule image:")
        print(f"  - Image type: {type(img)}")
        print(f"  - Image size: {img.size}")
        print(f"  - Image mode: {img.mode}")
        return img
    except Exception as e:
        print(f"[Visualization] ERROR creating molecule image:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        print(f"  - Error args: {e.args}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

def create_molecule_response(smiles, is_3d=False):
    """Create a response containing molecule visualization data."""
    print(f"\n[Visualization] Creating molecule response:")
    print(f"  - SMILES: {smiles}")
    print(f"  - 3D Mode: {is_3d}")
    
    try:
        # Convert SMILES to molecule
        print("[Visualization] Converting SMILES to RDKit molecule...")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("[Visualization] ERROR: Failed to create molecule from SMILES")
            print(f"  - Input SMILES: {smiles}")
            print("  - Possible issues:")
            print("    * Invalid SMILES syntax")
            print("    * Unsupported chemical features")
            print("    * Malformed input string")
            return None
            
        print("[Visualization] Successfully created RDKit molecule:")
        print(f"  - Canonical SMILES: {Chem.MolToSmiles(mol)}")
        print(f"  - Number of atoms: {mol.GetNumAtoms()}")
        print(f"  - Number of bonds: {mol.GetNumBonds()}")
        
        if is_3d:
            print("[Visualization] Preparing 3D response...")
            # For 3D, we'll send the SMILES string to be rendered by 3Dmol.js
            response = {
                            'smiles': smiles,
                            'is_3d': True,
                            'format': '3dmol',
                            'molecular_weight': "{:.2f}".format(MoleculeHandler().calculate_molecular_weight(mol))
            }
            print("[Visualization] Created 3D response object:")
            print(f"  - Response keys: {list(response.keys())}")
            
        else:
            print("[Visualization] Preparing 2D response...")
            # Generate 2D image
            img = create_molecule_image(mol)
            if img is None:
                print("[Visualization] ERROR: Failed to generate 2D molecule image")
                return None
                
            print("[Visualization] Converting image to base64...")
            # Convert image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            print(f"  - Base64 string length: {len(img_str)}")
            
            # Create response data
            response = {
                            'smiles': smiles,
                            'image': img_str,
                            'format': 'png',
                            'encoding': 'base64',
                            'molecular_weight': "{:.2f}".format(MoleculeHandler().calculate_molecular_weight(mol)),
                            'is_3d': False,
                            'image_size': f"{img.size[0]}x{img.size[1]}"
            }
            print("[Visualization] Created 2D response object:")
            print(f"  - Response keys: {list(response.keys())}")
            print(f"  - Image size: {response['image_size']}")
        
        print("[Visualization] Successfully generated molecule response")
        return response
        
    except Exception as e:
        print(f"\n[Visualization] ERROR creating molecule response:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        print(f"  - Generation mode: {'3D' if is_3d else '2D'}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

def generate_random_molecule():
    """Generate a random molecule for testing."""
    print("[Visualization] Generating random molecule")
    try:
        # List of common SMILES for testing
        test_molecules = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
            'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Salbutamol
            'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F',  # Celecoxib
            'CC1=C(C=C(C=C1)O)C(=O)CC2=CC=C(C=C2)OC',  # Nabumetone
            'CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3',  # Antipyrine
            'CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)CC3=CC=C(C=C3)OCC4=CC=CC=C4',  # Cinnarizine
        ]
        import random
        smiles = random.choice(test_molecules)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to create molecule from SMILES: {smiles}")
            
        print(f"[Visualization] Generated random molecule: {smiles}")
        return mol, smiles
        
    except Exception as e:
        print(f"[Visualization] Error generating random molecule: {str(e)}")
        print(f"[Visualization] Error type: {type(e)}")
        print(f"[Visualization] Error args: {e.args}")
        return None, None

def create_plot_response(data):
    """Create a response object for plot visualization."""
    try:
        # Create figure
        fig = go.Figure()
        
        if data.get('type') == '2d':
            # Create scatter plot for HSQC or COSY
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                marker=dict(
                    size=data.get('sizes', [10] * len(data['x'])),
                    color=data['z'],
                    colorscale=data.get('colorscale', 'Viridis'),
                    showscale=True,
                    opacity=0.7
                ),
                name='Correlations'
            ))
            
            # Reverse y-axis for NMR convention
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(autorange="reversed")
            
        else:  # 1D plot (1H or 13C NMR)
            # Main spectrum line
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines',
                line=dict(
                    color='rgb(0, 100, 200)',
                    width=1.5
                ),
                name='Spectrum'
            ))
            
            # Add vertical lines for peaks
            if 'peak_x' in data and 'peak_y' in data:
                fig.add_trace(go.Scatter(
                    x=data['peak_x'],
                    y=data['peak_y'],
                    mode='lines',
                    line=dict(
                        color='rgba(0, 100, 200, 0.5)',
                        width=1
                    ),
                    showlegend=False
                ))
            
            # Reverse x-axis for NMR convention
            fig.update_xaxes(autorange="reversed")
        
        # Update layout
        fig.update_layout(
            title=data.get('title', 'NMR Spectrum'),
            xaxis_title=data.get('x_label', 'Chemical Shift (ppm)'),
            yaxis_title=data.get('y_label', 'Intensity'),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(family="Arial", size=12),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray',
                zerolinewidth=1
            )
        )
        
        # Convert to JSON
        plot_json = fig.to_json()
        
        return {
            'plot': plot_json,
            'title': data.get('title', 'NMR Spectrum')
        }
        
    except Exception as e:
        print(f"Error creating plot response: {str(e)}")
        return None
