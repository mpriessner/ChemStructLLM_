import os
import io
import re
import ast
import math
import json
import pickle
import random
import logging
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import permutations
from typing import Dict, Any, List, Optional, Tuple
import requests
from typing import Dict, Any
import pandas as pd
import anthropic
from rdkit import Chem

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import cairosvg
from IPython.display import SVG

# Project-specific imports (assuming these are local modules)
import utils_MMT.extracting_data_KG_v15_4 as ed
import utils_MMT.data_generation_v15_4 as dg
import utils_MMT.plotting_v15_4 as pl
import utils_MMT.knowledge_graph_v15_4 as kg
import utils_MMT.similarity_functions_exp_v15_4 as sfe
#import config
# Anthropic/Claude API setup
import anthropic
from anthropic import Anthropic
from openai import OpenAI

#anthropic_api_key="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"
#client = anthropic.Anthropic(api_key=config.anthropic_api_key)


def save_variables(save_folder, variables_dict):
    """
    Save variables to a specified folder.
    
    :param save_folder: Path to the folder where variables will be saved
    :param variables_dict: Dictionary of variable names and their values to be saved
    """
    os.makedirs(save_folder, exist_ok=True)
    
    for var_name, var_value in variables_dict.items():
        file_path = os.path.join(save_folder, f"{var_name}.pkl")
        
        if isinstance(var_value, pd.DataFrame):
            var_value.to_pickle(file_path)
        elif isinstance(var_value, nx.Graph):
            nx.write_gpickle(var_value, file_path)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(var_value, f)
    
    print(f"Variables saved to {save_folder}")

def load_variables(save_folder):
    """
    Load variables from a specified folder.
    
    :param save_folder: Path to the folder where variables are saved
    :return: Dictionary of variable names and their loaded values
    """
    loaded_vars = {}
    
    for filename in os.listdir(save_folder):
        if filename.endswith(".pkl"):
            var_name = os.path.splitext(filename)[0]
            file_path = os.path.join(save_folder, filename)
            
            if var_name in ['df_13c', 'df_1h', 'df_hsqc', 'df_cosy']:
                loaded_vars[var_name] = pd.read_pickle(file_path)
            elif var_name in ['G', 'master_KG']:
                loaded_vars[var_name] = nx.read_gpickle(file_path)
            else:
                with open(file_path, 'rb') as f:
                    loaded_vars[var_name] = pickle.load(f)
    
    print(f"Variables loaded from {save_folder}")
    return loaded_vars


#### Remove side atom by index


from rdkit import Chem
from rdkit.Chem import AllChem

def remove_side_atom_by_index(smiles, index):
    """
    Removes an atom at the specified index from the molecule represented by the SMILES string.
    If the atom is part of a ring, the ring is made smaller by connecting the neighbors.
    
    Args:
    smiles (str): The input SMILES string of the molecule.
    index (int): The index of the atom to be removed (0-based).
    
    Returns:
    str: A new SMILES string with the specified atom removed, or None if the operation fails.
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES string")
        return None
    
    # Check if the index is valid
    if index < 0 or index >= mol.GetNumAtoms():
        print(f"Invalid index: {index}. Molecule has {mol.GetNumAtoms()} atoms.")
        return None
    
    # Create an editable molecule
    edit_mol = Chem.RWMol(mol)
    
    # Get the atom to be removed
    atom_to_remove = edit_mol.GetAtomWithIdx(index)
    
    # Get neighbors of the atom to be removed
    neighbors = atom_to_remove.GetNeighbors()
    
    # Check if the atom is in a ring
    is_in_ring = atom_to_remove.IsInRing()
    
    # Store neighbor indices and their aromatic status before removing the atom
    neighbor_info = [(n.GetIdx(), n.GetIsAromatic()) for n in neighbors]
    
    # Remove the atom
    edit_mol.RemoveAtom(index)
    
    # If the removed atom was in a ring or had exactly two neighbors, always connect the neighbors
    if is_in_ring or len(neighbor_info) == 2:
        # Sort neighbors to ensure consistent behavior
        neighbor_info.sort(key=lambda x: x[0])
        # Connect the neighbors with a bond that maintains aromaticity if possible
        if all(is_aromatic for _, is_aromatic in neighbor_info):
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = Chem.BondType.SINGLE
        edit_mol.AddBond(neighbor_info[0][0], neighbor_info[1][0], bond_type)
    
    # Convert back to mol and generate SMILES
    new_mol = edit_mol.GetMol()
    
    # Attempt to sanitize the molecule
    try:
        Chem.SanitizeMol(new_mol)
    except:
        print("Warning: Failed to sanitize the molecule. Attempting to create a valid structure.")
        # If sanitization fails, try to create a valid structure by setting all bonds to single
        for bond in new_mol.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(new_mol)
        except:
            print("Failed to create a valid structure.")
            return None
    
    # Generate the new SMILES
    new_smiles = Chem.MolToSmiles(new_mol)
    return new_smiles


def is_end_group(mol, atom_index):
    """
    Checks if the atom at the given index is an end-group or terminal atom.
    
    Args:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
    atom_index (int): The index of the atom to check (0-based).
    
    Returns:
    bool: True if the atom is an end-group, False otherwise.
    """
    if atom_index < 0 or atom_index >= mol.GetNumAtoms():
        raise ValueError(f"Invalid atom index: {atom_index}")
    
    atom = mol.GetAtomWithIdx(atom_index)
    
    # Check if the atom has only one neighbor (excluding hydrogens)
    if atom.GetDegree() == 1:
        # It's an end-group if it's not a hydrogen
        return atom.GetAtomicNum() != 1
    
    # Special case for methyl groups: carbon with 3 hydrogens
    if atom.GetAtomicNum() == 6 and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 3:
        return True
    
    # Special case for aldehydes: carbon with one double-bonded oxygen and one single bond
    if atom.GetAtomicNum() == 6 and atom.GetDegree() == 2:
        neighbors = atom.GetNeighbors()
        if any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom_index, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE for n in neighbors):
            return True
    
    # Special case for terminal oxygen in hydroxyl, ether, or ester groups
    if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1:
        return True
    
    return False


def filter_fragments_list(fragment_list, target_connections):
    """
    Filter a list of fragment dictionaries to keep only the entries with the specified number of submol_connection_points.
    
    :param fragment_list: List of dictionaries containing fragment data
    :param target_connections: Number of connection points to filter for
    :return: List of updated fragment dictionaries
    """
    def filter_fragment_connections(fragment_dict, target_connections):
        """
        Filter the fragment dictionary to keep only the entries with the specified number of submol_connection_points.
        
        :param fragment_dict: Dictionary containing fragment data
        :param target_connections: Number of connection points to filter for
        :return: Updated fragment dictionary
        """
        # Find indices of submol_connection_points_list with the target number of connections
        valid_indices = [i for i, points in enumerate(fragment_dict['submol_connection_points_list']) 
                         if len(points) == target_connections]
        
        # Update lists in the dictionary
        for key in ['submol_connection_points_list', 'mol_connection_points_list', 
                    'connection_mapping_list', 'submols']:
            fragment_dict[key] = [fragment_dict[key][i] for i in valid_indices]
        
        # Update occurrences
        fragment_dict['occurrences'] = len(valid_indices)
        
        return fragment_dict

    # Create a new list to store filtered fragments
    filtered_fragments = []

    # Iterate over each fragment in the list
    for fragment in fragment_list:
        # Apply filtering to each fragment
        filtered_fragment = filter_fragment_connections(fragment, target_connections)
        
        # Append the filtered fragment to the new list
        filtered_fragments.append(filtered_fragment)

    return filtered_fragments

# Usage example with a list of fragments
fragments = [
    {
        'id': 'F15',
        'smiles': 'cC(C)C',
        'weight': 46.028,
        'occurrences': 16,
        'avg_c13_shift': 24.879296531250002,
        'avg_h1_shift': 3.18454066875,
        'avg_hsqc_data': (24.879296531250002, 3.206791725),
        'num_open_connections': 2,
        'submol_connection_points_list': [[0, 2, 3], [0, 2, 3], [0, 2, 3], [0, 2, 3], [0], [0], [0], [0, 2, 3], [0, 2, 3], [0], [0], [0], [0, 2, 3], [0, 2, 3], [0, 2, 3], [0]],
        'mol_connection_points_list': [[12, 14, 15], [13, 15, 25], [20, 22, 23], [14, 16, 19], [19], [3], [15], [20, 22, 23], [23, 25, 26], [7], [17], [8], [2, 4, 5], [5, 7, 22], [13, 15, 18], [16]],
        'connection_mapping_list': [{0: [11, 16]}, {0: [12, 26], 2: [16], 3: [18]}, {0: [19, 16]}, {0: [13, 9], 2: [17, 18], 3: [20]}, {0: [18, 15]}, {0: [2, 7]}, {0: [14, 19]}, {0: [19, 24]}, {0: [7, 4]}, {0: [6, 11]}, {0: [16, 21]}, {0: [7, 12]}, {0: [1, 6]}, {0: [4, 23], 2: [8], 3: [21]}, {0: [12, 19], 2: [16], 3: [17]}, {0: [15, 4]}],
        'submols': [object()] * 16  # Placeholder for RDKit Mol objects
    },
    {
        'id': 'F190',
        'smiles': 'CC(C)[NH3+]',
        'weight': 47.016,
        'occurrences': 3,
        'avg_c13_shift': 50.56337033333333,
        'avg_h1_shift': 3.7389272333333334,
        'avg_hsqc_data': (50.56337033333333, 3.7610439),
        'num_open_connections': 2,
        'submol_connection_points_list': [[0, 3], [0, 3], [0, 3]],
        'mol_connection_points_list': [[3, 6], [9, 12], [14, 17]],
        'connection_mapping_list': [{0: [2], 3: [7, 8]}, {0: [8], 3: [13, 14, 8]}, {0: [13], 3: [18]}],
        'submols': [object()] * 3  # Placeholder for RDKit Mol objects
    }
]

# Filter fragments with exactly 3 connection points
filtered_fragments = filter_fragments_list(fragments, 3)

# Output the filtered results for verification
for i, filtered_fragment in enumerate(filtered_fragments):
    print(f"Fragment {i+1} ID: {filtered_fragment['id']}")
    print(f"Occurrences after filtering: {filtered_fragment['occurrences']}")
    print(f"Filtered submol_connection_points_list length: {len(filtered_fragment['submol_connection_points_list'])}")
    if filtered_fragment['submol_connection_points_list']:
        print(f"Example of filtered submol_connection_points: {filtered_fragment['submol_connection_points_list'][0]}")
    print("-" * 40)


#### Replacement Version 4


def select_substructure(mol, center_atom_idx, radius):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom_idx)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    
    submol_connection_points = []
    mol_connection_points = []
    connection_mapping = {}
    reverse_amap = {v: k for k, v in amap.items()}
    
    for submol_idx in range(submol.GetNumAtoms()):
        if submol_idx in reverse_amap:
            orig_idx = reverse_amap[submol_idx]
            submol_atom = submol.GetAtomWithIdx(submol_idx)
            orig_atom = mol.GetAtomWithIdx(orig_idx)
            if submol_atom.GetDegree() < orig_atom.GetDegree():
                submol_connection_points.append(submol_idx)
                mol_connection_points.append(orig_idx)
                for neighbor in orig_atom.GetNeighbors():
                    if neighbor.GetIdx() not in reverse_amap.values():
                        connection_mapping[submol_idx] = connection_mapping.get(submol_idx, []) + [neighbor.GetIdx()]
    
    num_open_connections = len(submol_connection_points)
    
    return submol, amap, submol_connection_points, mol_connection_points, connection_mapping, num_open_connections


    
def adjust_atom_valence(rwmol, atom_idx):
    if atom_idx >= rwmol.GetNumAtoms():
        return rwmol  # Return unchanged if atom doesn't exist
    
    atom = rwmol.GetAtomWithIdx(atom_idx)
    atom.UpdatePropertyCache(strict=False)
    
    if atom.GetSymbol() == 'S':
        max_valence = 6
        current_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        
        if current_valence > max_valence:
            valence_to_reduce = current_valence - max_valence
            
            # First, remove H atoms if present and not needed
            h_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if h_neighbors and valence_to_reduce > 0:
                rwmol.RemoveAtom(h_neighbors[0].GetIdx())
                valence_to_reduce -= 1
            
            # Remove Explicit hydrogens
            total_h = atom.GetNumExplicitHs()
            h_to_remove = min(valence_to_reduce, total_h)
            atom.SetNumExplicitHs(max(0, total_h - h_to_remove))
            valence_to_reduce -= h_to_remove
            #print(f"Removed {h_to_remove} hydrogens. New total H: {atom.GetNumExplicitHs()}")
                
            # If we still need to reduce valence, change bond orders
            if valence_to_reduce > 0:
                for bond in atom.GetBonds():
                    if valence_to_reduce == 0:
                        break
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        rwmol.RemoveBond(atom.GetIdx(), bond.GetOtherAtomIdx(atom.GetIdx()))
                        rwmol.AddBond(atom.GetIdx(), bond.GetOtherAtomIdx(atom.GetIdx()), Chem.BondType.SINGLE)
                        valence_to_reduce -= 1
    
    atom.UpdatePropertyCache(strict=False)
    #print(f"Final valence of atom {atom_idx}: {atom.GetTotalValence()}")
    return rwmol

def show_mols(mols, mols_per_row=2, size=300, min_font_size=12, legends=[], file_name=''):
    if legends and len(legends) < len(mols):
        print('legends is too short')
        return None

    mols_per_row = min(len(mols), mols_per_row)
    rows = math.ceil(len(mols)/mols_per_row)
    d2d = rdMolDraw2D.MolDraw2DSVG(mols_per_row*size, rows*size, size, size)
    d2d.drawOptions().minFontSize = min_font_size
    if legends:
        d2d.DrawMolecules(mols, legends=legends)
    else:
        d2d.DrawMolecules(mols)
    d2d.FinishDrawing()

    if file_name:
        with open('d2d.svg', 'w') as f:
            f.write(d2d.GetDrawingText())
        if file_name.endswith('.pdf'):
            cairosvg.svg2pdf(url='d2d.svg', write_to=file_name)
        else:
            cairosvg.svg2png(url='d2d.svg', write_to=file_name)
        os.remove('d2d.svg')

    return SVG(d2d.GetDrawingText())



def combine_connection_mappings(connection_mapping_sub, connection_mapping_target):
    new_mapping = {}
    sub_keys = list(connection_mapping_sub.keys())
    target_values = list(connection_mapping_target.values())
    
    # Ensure we have enough values from the target mapping
    if len(sub_keys) > len(target_values):
        raise ValueError("Not enough values in target mapping to match all keys in sub mapping")
    
    # Combine keys from sub and values from target
    for i, key in enumerate(sub_keys):
        new_mapping[key] = target_values[i]
    
    return new_mapping


def get_all_combinations(points):
    return list(permutations(points))

def index_preserving_replace_substructure(mol, center_atom_idx, radius, replacement_sub_mol, connection_mapping_sub, output_path):
    # Make all hydrogens explicit
    #mol = Chem.AddHs(mol)

    # Select the substructure
    submol, amap, submol_connection_points_target, mol_connection_points, connection_mapping_target, _ = select_substructure(mol, center_atom_idx, radius)
    
    #submol_connection_points = extend_list_based_on_dict(connection_mapping, submol_connection_points)

    # Ensure replacement_mol has explicit hydrogens
    #replacement = Chem.AddHs(replacement_mol)
    if replacement_sub_mol is None:
        pass

    
    # Reverse the amap dictionary
    reversed_amap = {v: k for k, v in amap.items()}

 # Store bond information before removal
    bond_info = {}
    bond_list = []
    for submol_idx, orig_idxs in connection_mapping_target.items():
        for orig_idx in orig_idxs:
            
            bond = mol.GetBondBetweenAtoms(reversed_amap[submol_idx], orig_idx)
            if bond:
                bond_info[orig_idx] = (reversed_amap[submol_idx], bond.GetBondType())
                bond_list.append(bond.GetBondType())
                
    # Create a new molecule with atoms from the original molecule not in the substructure
    new_mol = Chem.RWMol(Chem.Mol())
    old_idx_to_new_idx = {}
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in reversed_amap.values():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_idx = new_mol.AddAtom(new_atom)

            # Store the original index as a property for reference
            new_mol.GetAtomWithIdx(new_idx).SetProp("old_idx", str(atom.GetIdx()))

            old_idx_to_new_idx[atom.GetIdx()] = new_idx

    # Add bonds between the remaining atoms
    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin_idx in old_idx_to_new_idx and end_idx in old_idx_to_new_idx:
            new_mol.AddBond(old_idx_to_new_idx[begin_idx], old_idx_to_new_idx[end_idx], bond.GetBondType())

    replacement_offset = new_mol.GetNumAtoms()
    
    # Add the replacement structure
    for atom in replacement_sub_mol.GetAtoms():
        #print(atom.GetSymbol)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_mol.AddAtom(new_atom)

    for bond in replacement_sub_mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        new_mol.AddBond(replacement_offset + begin_idx, replacement_offset + end_idx, bond.GetBondType())

    combined_mapping = combine_connection_mappings(connection_mapping_sub, connection_mapping_target)
        
    idx = 0 
    # Connect the replacement to the original molecule using connection_mapping
    for submol_idx, orig_idxs in combined_mapping.items():
        for orig_idx in orig_idxs:
            if orig_idx in old_idx_to_new_idx:  # Check if the original atom still exists
                #bond_type = bond_info.get(idx, (None, Chem.BondType.SINGLE))[1]
                bond_type = bond_list[idx]
                #bond_type = bond_info_list[idx]
                new_orig_idx = old_idx_to_new_idx[orig_idx]
                new_replacement_idx = replacement_offset + submol_idx
                idx += 1
                new_mol.AddBond(new_orig_idx, new_replacement_idx, bond_type)
                #print(new_orig_idx, new_replacement_idx, bond_type)
    #import IPython; IPython.embed();

    # Get the final molecule
    final_mol = new_mol.GetMol()
    try:
        final_mol = Chem.RemoveHs(final_mol)
        final_mol = Chem.AddHs(final_mol)    
        smi = Chem.MolToSmiles(final_mol)
        print(smi)
    except Chem.KekulizeException:
        for atom in final_mol.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in final_mol.GetBonds():
            bond.SetIsAromatic(False)
    
    rwmol = Chem.RWMol(final_mol)
    for atom in rwmol.GetAtoms():
        rwmol = Chem.RWMol(adjust_atom_valence(rwmol, atom.GetIdx()))

    final_mol = rwmol.GetMol()

    # Update property cache for all atoms before adjusting valences
    for atom in final_mol.GetAtoms():
        atom.UpdatePropertyCache(strict=False)

    # Adjust valences for all atoms in the final molecule
    for atom in final_mol.GetAtoms():
        adjust_atom_valence(final_mol, atom.GetIdx())

    try:
        Chem.SanitizeMol(final_mol)
    except Chem.KekulizeException:
        for atom in final_mol.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in final_mol.GetBonds():
            bond.SetIsAromatic(False)
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    AllChem.Compute2DCoords(final_mol)
    AllChem.Compute2DCoords(submol)
    AllChem.Compute2DCoords(replacement_sub_mol)

    # Draw the molecules
    mols = [mol, final_mol, submol, replacement_sub_mol]
    legends = ["Original", "Modified", "Substructure", "Replacement"]
    #import IPython; IPython.embed();
    svg = show_mols(mols, mols_per_row=2, size=300, legends=legends, file_name=output_path)
    
    return final_mol, output_path, svg



#### Extract information for filtering
def extract_information(text):
    """
    Extracts numbers following "REVISIT:" from the given text, distinguishing between
    regular REVISIT and REVISIT (remove) cases, and extracts the QUALITY score in percentage format.
    
    Args:
    text (str): The input text to search for REVISIT numbers and QUALITY score.
    
    Returns:
    tuple: Three elements representing the extracted information:
           (revisit_numbers, remove_numbers, quality_score)
           quality_score is returned as a float between 0 and 100
    """
    import re
    
    revisit_numbers = []
    remove_numbers = []
    quality_score = None
    
    # Process the text line by line
    for line in text.split('\n'):
        # Pattern for REVISIT numbers
        revisit_match = re.match(r'^REVISIT:\s*(\d+)\s*$', line.strip())
        if revisit_match:
            revisit_numbers.append(int(revisit_match.group(1)))
            continue
        
        # Pattern for REVISIT (remove) numbers
        remove_match = re.match(r'^REVISIT:\s*(\d+)\s*\(remove\)\s*$', line.strip())
        if remove_match:
            remove_numbers.append(int(remove_match.group(1)))
            continue
        
        # Pattern for QUALITY score in percentage format
        quality_match = re.match(r'^QUALITY:\s*(\d+\.?\d*)\s*%\s*$', line.strip())
        if quality_match:
            quality_score = float(quality_match.group(1))
            # Ensure score is within valid range (0-100)
            quality_score = min(max(quality_score, 0), 100)
    
    return revisit_numbers, remove_numbers, quality_score


def extract_nmr_data(text, nmr_type):
    prefix = f"__{nmr_type}__: "
    start = text.find(prefix)
    if start != -1:
        start += len(prefix)
        end = text.find("\n", start)
        if end == -1:  # If it's the last line
            end = len(text)
        data_str = text[start:end].strip()
        if data_str == 'None':
            return None
        try:
            return ast.literal_eval(data_str)
        except (ValueError, SyntaxError):
            return data_str  # Return as string if not a valid literal
    return None

def generate_filter_dict(summary, hsqc_delta=10, c13_delta=10, h1_delta=10):
    HSQC_suggestion = extract_nmr_data(summary, "HSQC")
    C13_suggestion = extract_nmr_data(summary, "13C")
    H1_suggestion = extract_nmr_data(summary, "1H")
    filters = {}
    if HSQC_suggestion and isinstance(HSQC_suggestion, list) and len(HSQC_suggestion) == 2:
        filters['hsqc_shift_range'] = (HSQC_suggestion[0], HSQC_suggestion[1], hsqc_delta)
    if C13_suggestion is not None:
        filters['c13_shift_range'] = (C13_suggestion, c13_delta)
    if H1_suggestion is not None:
        filters['h1_shift_range'] = (H1_suggestion, h1_delta)
    return filters

def extract_fragment_info(filtered_has_fragment):
    if not filtered_has_fragment:
        return None, None, None
    #import IPython; IPython.embed();
    
    fragment_data = filtered_has_fragment[0][2][1]  # Access the fragment data dictionary
    weight = fragment_data.get('molecular_weight')
    connection_mapping = fragment_data.get('connection_mapping')
    connection_sub = len(connection_mapping.keys())
    connection_core_list = connection_mapping.values()                           
    connection_core = len(
        [item for sublist in connection_core_list 
         for item in (sublist if isinstance(sublist, list) else [sublist])]
    )        
    return weight, connection_core, connection_sub

def update_filters_with_fragment_info(filters, weight, connection_core, connection_sub, weight_delta=5):
    if weight is not None:
        filters['weight_range'] = (weight - weight_delta, weight + weight_delta)
    if connection_core is not None:
        filters['num_connection_points_core'] = [connection_core]
    if connection_sub is not None:
        filters['num_connection_points_sub'] = [connection_sub]
    return filters


def get_connection_mapping_counts_revisit(revisit_filtered_fragment):
    if not revisit_filtered_fragment:
        return None

    try:
        fragment_data = revisit_filtered_fragment[0][2][1]  # Access the fragment data dictionary
        connection_mapping = fragment_data.get('connection_mapping')
        
        if connection_mapping:
            return [len(values) for values in connection_mapping.values()]
        else:
            return None
    except (IndexError, KeyError, AttributeError):
        return None

def prepare_search_filters(summary, revisit_filtered_fragment, weight_delta=15, hsqc_delta=10, c13_delta=10, h1_delta=10):
    # Generate initial filters from NMR data
    filters = generate_filter_dict(summary, hsqc_delta, c13_delta, h1_delta)
    
    # Extract fragment info
    weight, num_open_connections_core, num_open_connections_sub = extract_fragment_info(revisit_filtered_fragment)
    
    # Update filters with fragment info
    filters = update_filters_with_fragment_info(filters, weight, num_open_connections_core, num_open_connections_sub, weight_delta)
    
    # Add new filter for connection mapping counts
    connection_mapping_counts = get_connection_mapping_counts_revisit(revisit_filtered_fragment)
    if connection_mapping_counts:
        filters['connection_mapping_counts'] = connection_mapping_counts
    
    return filters

def filter_has_fragment(graph_data, atom_index):
    filtered_fragment = []
    
    for item in graph_data:
        has_fragment_relations = item[2]
        for relation in has_fragment_relations:
            relation_type, molecule_data, fragment_data, relation_info = relation
            
            # Access the fragment data dictionary, which is the third element in the tuple
            fragment_info = fragment_data[1]
            
            if relation_type == 'HAS_FRAGMENT' and fragment_info['main_molecule_atom_index'] == atom_index:
                filtered_fragment.append(relation)
    
    return filtered_fragment



def plot_molecule(smiles, figsize=(4, 4)):
    """
    Plot a molecule given its SMILES string.
    
    Args:
    smiles (str): SMILES string of the molecule
    figsize (tuple): Figure size (width, height) in inches
    
    Returns:
    None
    """
    # Create an RDKit molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate a 2D depiction
    img = Draw.MolToImage(mol)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Display the image using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(plt.imread(io.BytesIO(img_byte_arr)))
    plt.axis('off')
    plt.title(f"Molecule: {smiles}")
    plt.show()

    
def custom_log(log_file:str, message: str):
    """
    Custom logging function that writes directly to the log file.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} - {message}\n"
    try:
        with open(log_file, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error writing to log file: {e}")
        
def log_conversation(log_file:str, agent: str, role: str, content: str):
    """
    Log a conversation entry using the custom logging function.
    """
    message = f"{agent} - {role}: {content}"
    custom_log(log_file, message)

# Example usage
#custom_log("Logging system initialized")
#log_conversation("Orchestration Agent", "system", "System prompt content here")
   
    

def make_api_call(client, api_type: str, request_body: dict=None, prompt:str=None, headers:dict=None, max_retries: int = 3, retry_delay: int = 5) -> dict:
    """
    Make an API call with automatic retries for both OpenAI and Claude APIs.
    
    Args:
        client: The API client (either OpenAI or Anthropic client)
        request_body: The request payload
        api_type: String indicating the API type ("openai" or "claude")
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 5)
        
    Returns:
        dict: The API response data
        
    Raises:
        Exception: If all retry attempts fail
    """
    import time
    from requests.exceptions import RequestException
    import logging
    
    attempt = 0
    last_exception = None
    
    while attempt < max_retries:
        try:
            if api_type.lower() == "openai":
                response = client.chat.completions.create(
                            model="o1-mini",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                                    ]
                                )  
                # For OpenAI, construct a response similar to Claude's format
                if type( response.choices[0].message.content) == str:
                    return response
                else:
                    attempt += 1            
                    time.sleep(retry_delay)

            elif api_type.lower() == "claude":
                # For Claude, we need to make the POST request
               
                response = client._client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=request_body,
                    headers=headers
                )
                #import IPython; IPython.embed();                
                if response.status_code == 200:
                    return response
                else:
                    attempt += 1
                    time.sleep(retry_delay)


            elif api_type.lower() == "claude_prompt":


                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2048,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                if type(response.content)==str:
                    return response.content
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
            
        except Exception as e:
            last_exception = e
            attempt += 1
            
            if attempt < max_retries:
                logging.warning(f"API call attempt {attempt} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"All retry attempts failed for {api_type} API call: {str(e)}")
                raise Exception(f"Failed to get response after {max_retries} attempts. Last error: {str(last_exception)}")
    
    # This should never be reached due to the raise in the loop, but adding for completeness
    raise Exception(f"Failed to get response after {max_retries} attempts. Last error: {str(last_exception)}")



def spectrum_nmr_agent(config:dict, task_type: str, image_paths: Dict[str, str], prediction: str = "", additional_feedback: str = "") -> Optional[str]:
    try:
        # Check if in test mode
        if hasattr(config, 'mode') and config.mode == "Test":
            return """This is a test \n FINISHED \nResults: PASS """

        # Encode images
        image1_data = pl.encode_image(image_paths['normal_view'])
        image2_data = pl.encode_image(image_paths['rotated_view'])
        
        prompt_file_path = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/spectrum_nmr_agent_system_prompt.txt"
        
        # Read the content of the system prompt file
        try:
            with open(prompt_file_path, 'r') as file:
                system_prompt = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The system prompt file was not found at {prompt_file_path}")
        except Exception as e:
            raise Exception(f"An error occurred while reading the system prompt file: {str(e)}")
        # Prepare the request body
        if task_type == "prediction":
            request_body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4096,
                "system": [
                    {
                        "type": "text",
                        "text": "You are an NMR prediction agent."
                    },
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image1_data
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image2_data
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Analyze the molecule structure images for getting the 1H, 13C, HSQC and COSY NMR spectra." + get_task_prompt(task_type, prediction, additional_feedback)
                            }
                        ]
                    }
                ]
            }
        else:
            request_body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 2048,
                "system": [
                    {
                        "type": "text",
                        "text": "You are an criticism agent."
                    },
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image1_data
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image2_data
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Analyze the molecule structure images for getting the 1H, 13C, HSQC and COSY NMR spectra." + get_task_prompt(task_type, prediction, additional_feedback)
                            }
                        ]
                    }
                ]
            }
        #import IPython; IPython.embed();

        # Prepare headers with API key and beta feature flag
        headers = {
            "x-api-key": config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json"
        }
        #print("spectrum agent")
        #import IPython; IPython.embed();
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)        
        # Make the API call with the custom headers

        response = make_api_call(client, "claude", request_body=request_body, headers=headers)

        # response = client._client.post(
        #     "https://api.anthropic.com/v1/messages",
        #     json=request_body,
        #     headers=headers
        # )
        response.raise_for_status()  # This will raise an exception for HTTP error responses
        
        response_data = response.json()
        print(f"\nUsage statistics for {task_type} task:")
        print(f"Input tokens: {response_data['usage']['input_tokens']}")
        print(f"Output tokens: {response_data['usage']['output_tokens']}")
        print(f"Cache creation input tokens: {response_data['usage'].get('cache_creation_input_tokens', 0)}")
        print(f"Cache read input tokens: {response_data['usage'].get('cache_read_input_tokens', 0)}")
        print("\n" + "-"*50 + "\n")
        return response_data['content'][0]['text'] if response_data['content'][0]['type'] == 'text' else str(response_data['content'])
    except Exception as e:
        logging.error(f"Error in spectrum_nmr_agent: {str(e)}")
        import IPython; IPython.embed();
        return response_data['content'][0]['text'] if response_data['content'][0]['type'] == 'text' else str(response_data['content'])
    
    
def get_task_prompt(task_type: str, prediction: str = "", additional_feedback: str = "") -> str:
    base_path = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/"
    
    if task_type == "prediction":
        prompt_file_path = f"{base_path}prediction_agent_system_prompt.txt"
    elif task_type == "criticism":
        prompt_file_path = f"{base_path}criticism_agent_system_prompt.txt"
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    try:
        with open(prompt_file_path, 'r') as file:
            task_prompt = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The task prompt file was not found at {prompt_file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the task prompt file: {str(e)}")

    if task_type == "prediction":
        return f"Predict the expected 1H, 13C, HSQC and COSY NMR peaks for this molecular structure. {task_prompt} Provide a detailed analysis, referencing the molecular structure and atom labels when relevant. Here is the feedback from the criticism agent: {additional_feedback}"
    else:  # criticism
        return f"Critically evaluate the following 1H, 13C, HSQC and COSY NMR peak prediction.\n\n {task_prompt} The prediction from prediction agent are as follows:\n\n{prediction}\n\n."

  

def get_valid_agent_response(config, task, image_paths, additional_feedback=None, prediction=None):
    max_retries = 3
    retries = 0
    accumulated_response = ""
    
    while retries < max_retries:
        # On first try, use original parameters
        if retries == 0:
            if task == "prediction":
                response = spectrum_nmr_agent(config, task, image_paths, additional_feedback=additional_feedback)
            elif task == "criticism":
                response = spectrum_nmr_agent(config, task, image_paths, prediction=prediction)
            else:
                raise ValueError(f"Unknown task: {task}")
        else:
            # On subsequent tries, include previous partial response as context
            continuation_prompt = (
                f"Previous partial analysis:\n{accumulated_response}\n"
                "Please continue the analysis from where it left off, ensuring all required information "
                "is included and end with FINISHED when complete."
            )
            
            if task == "prediction":
                response = spectrum_nmr_agent(config, task, image_paths, 
                                           additional_feedback=continuation_prompt)
            elif task == "criticism":
                response = spectrum_nmr_agent(config, task, image_paths, 
                                           prediction=prediction,
                                           additional_feedback=continuation_prompt)
        
        if response is not None:
            # Add the new response to accumulated response
            if accumulated_response:
                # Remove any duplicated content from the beginning of the new response
                # This helps avoid repeating the structural analysis part
                accumulated_response = accumulated_response + "\n\n" + response
                #accumulated_response += "\n" + new_content
            else:
                accumulated_response = response

            # Check if we have a complete response
            if "FINISHED" in accumulated_response:
                return accumulated_response
        
        retries += 1
        print(f"{task.capitalize()} incomplete. Retry {retries}/{max_retries}... "
              f"Accumulated length: {len(accumulated_response)}")
    
    raise Exception(f"Failed to get valid {task} after {max_retries} attempts. "
                   f"Partial response: {accumulated_response}")
        

def send_to_specialized_agents(config:dict, nmr_data: Dict[str, Any], image_paths: Dict[str, str], agent_types: List[str]) -> Dict[str, str]:
    results = {}
    log_file = config.log_file
    final_prediction = ""
    criticism_ticker = 0
    criticism = ""
    additional_feedback = ""
    additional_feedback_cutout = ""    

    while True and final_prediction=="":
        try:
            # Call prediction agents
            prediction_1 = get_valid_agent_response(config, "prediction", image_paths, additional_feedback)
            log_conversation(log_file, f"NMR_Agent", "Prediction agent 1", prediction_1)

            # Combine predictions
            prediction = "\n\n First prediction agent:\n " + prediction_1 

            # Call criticism agent
            criticism = get_valid_agent_response(config, "criticism", image_paths, prediction=prediction)
            log_conversation(log_file, f"NMR_Agent", "Criticism", criticism)
            #import IPython; IPython.embed();

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            #import IPython; IPython.embed();
        
        if "Results: PASS" in criticism or criticism_ticker == 2:  ######## CHANGE"""
            print(f"Prediction: Pass")
            #import IPython; IPython.embed();

            # Extract ADDITIONAL FEEDBACK for PASS case
            if "__ADDITIONAL FEEDBACK__:" in criticism:
                additional_feedback = criticism.split("__ADDITIONAL FEEDBACK__:")[1].strip()
            
            # Combine prediction and additional feedback
            final_prediction = f"{prediction}\n\nADDITIONAL FEEDBACK:\n{additional_feedback}".strip()
            break
            
        elif "Results: FAIL" in criticism:
            print(f"Prediction: Fail")
            
            # Extract feedback for FAIL case
            #if "__ADDITIONAL FEEDBACK__:" in criticism:
                #additional_feedback_cutout = criticism.split("__ADDITIONAL FEEDBACK__:")[1].strip()
            additional_feedback = f"Dear NMR prediction agent. Our inependent analysis identified some errors in your NMR analysis. Your previous prediction was as follows: {prediction}\n\n The feedback from the Criticism Agent is as follows: {criticism} \n\n Please revise your predictions based on the new information provided and correct it if necessary."
            #print(f"Rerunning prediction with feedback: {additional_feedback}")
            criticism_ticker += 1
            #import IPython; IPython.embed();

        else:
            print(f"Unexpected criticism result for prediction")
            #raise ValueError(f"Unexpected criticism result for prediction")
            import IPython; IPython.embed();
    # Call the final analysis with the combined prediction and feedback
    
    results["final_prediction"] = final_prediction
      
    for agent_type in agent_types:
        custom_log(log_file, f"###################{agent_type.upper()}###################")

        if agent_type.upper() in nmr_data:
            while True:
                try:
                    #response = call_agent(config, agent_type, nmr_data[agent_type.upper()], f"Analyze {agent_type.upper()} data", image_paths, final_prediction)
                    response = call_agent_o1(config, agent_type, nmr_data[agent_type.upper()], f"Analyze {agent_type.upper()} data", final_prediction)
                    #response = spectrum_nmr_agent("analysis", agent_type, image_paths, prediction=combined_prediction)
                    results[f'{agent_type.upper()}_analysis'] = response

                    # Log the response from each specialized agent
                    log_conversation(log_file, f"{agent_type.upper()} Agent", "assistant", response)
                    if type(response) == str:
                        break
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                
    custom_log(config.log_file, "Sent data to specialized agents and received results")
    return results



def generate_shifts_batch(config, smiles_list):
    output_directory = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/trash_folder"
    csv_file_path = ed.save_smiles_as_csv(smiles_list, output_directory)
    custom_log(config.log_file, f"CSV file saved at: {csv_file_path}")

    config.SGNN_csv_gen_smi = csv_file_path
    df_gen_data = dg.run_sgnn(config)
    
    list_nmr_data = []
    list_nmr_data_all = []
    list_sdf_paths = []
    list_smi = []
    for _, row in df_gen_data.iterrows():

        try:
            sdf_path = row['sdf_path']
            list_sdf_paths.append(sdf_path)

            # Generate NMR data
            df_hsqc, df_hsqc_all, _ = ed.run_HSQC_generation(sdf_path)
            df_cosy, _ = ed.run_COSY_generation(sdf_path)
            df_13c, df_13c_all = ed.run_13C_generation(sdf_path)
            #import IPython; IPython.embed();
            df_1h = ed.run_1H_generation(sdf_path)
    
            nmr_data = {
                'HSQC': df_hsqc.to_dict(),
                'COSY': df_cosy.to_dict(),
                '13C': df_13c.to_dict(),
                '1H': df_1h.to_dict()
            }
        
            nmr_data_all = {
                'HSQC': df_hsqc_all.to_dict(),
                'COSY': df_cosy.to_dict(),
                '13C': df_13c_all.to_dict(),
                '1H': df_1h.to_dict()
            }
            
            mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
            smiles = Chem.MolToSmiles(mol)

            list_nmr_data.append(nmr_data)
            list_nmr_data_all.append(nmr_data_all)
            list_smi.append(smiles)            
        except:
            print(row)

    return list_nmr_data, list_nmr_data_all, list_sdf_paths, list_smi


def add_atom_index_column(df):
    """
    Adds an 'atom_index' column to the DataFrame if it doesn't exist,
    copying values from the 'Self_Index' column.
    
    Args:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: DataFrame with 'atom_index' column added if necessary
    """
    try:
        if 'atom_index' not in df.columns and 'Self_Index' in df.columns:
            df['atom_index'] = df['Self_Index']
    except:
        if 'atom_index' not in df.columns and 'parent_atom_index' in df.columns:
            df['atom_index'] = df['parent_atom_index']
    return df

    
def compare_nmr_data(results):
    ##### MAKE IT LATE POSSIBEL TO RUN ALSO WITHOUT ALL DATA INPUTS e.g. just HSQC #######
    # Apply this function to all your DataFrames
    df_HSQC_gen_ = add_atom_index_column(pd.DataFrame(results['guess_nmr_data']["HSQC"]))
    df_HSQC_trg_ = add_atom_index_column(pd.DataFrame(results['target_nmr_data']["HSQC"]))
    df_COSY_gen_ = add_atom_index_column(pd.DataFrame(results['guess_nmr_data']["COSY"]))
    df_COSY_trg_ = add_atom_index_column(pd.DataFrame(results['target_nmr_data']["COSY"]))

    overall_error_HSQC, df_HSQC_gen_, df_HSQC_trg_ = sfe.similarity_calculations_HSQC(df_HSQC_gen_, df_HSQC_trg_, mode="hung_dist_nn", similarity_type="euclidean", error="sum", assignment_plot=False)
    overall_error_COSY, df_COSY_gen_, df_COSY_trg_ = sfe.similarity_calculations_COSY(df_COSY_gen_, df_COSY_trg_, mode="hung_dist_nn", similarity_type="euclidean", error="sum", assignment_plot=False)

    df_HSQC_gen_ = df_HSQC_gen_.to_dict()  
    df_HSQC_trg_ = df_HSQC_trg_.to_dict() 
    df_COSY_gen_ = df_COSY_gen_.to_dict() 
    df_COSY_trg_ = df_COSY_trg_.to_dict() 
    
    results['guess_nmr_data']["HSQC"] = {key: df_HSQC_gen_[key] for key in ['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index']}
    results['target_nmr_data_']["HSQC"] = {key: df_HSQC_trg_[key] for key in ['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index']}
    
    results['guess_nmr_data']["COSY"] = {key: df_COSY_gen_[key] for key in ['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index']}
    results['target_nmr_data_']["COSY"] = {key: df_COSY_trg_[key] for key in ['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index']}
    
    results['guess_nmr_data']["13C"] = {key: results['guess_nmr_data']["13C"][key] for key in ['shifts', 'atom_index']}
    results['target_nmr_data_']["13C"] = {key: results['target_nmr_data_']["13C"][key] for key in ['shifts', 'atom_index']}

    results['guess_nmr_data']["1H"] = {key: results['guess_nmr_data']["1H"][key] for key in ['shifts', 'intensity', "parent_atom_indices", "multiplicity_patterns_and_num_hydrogens", "shifts_orig"]}
    results['target_nmr_data_']["1H"] = {key: results['target_nmr_data_']["1H"][key] for key in ['shifts', 'intensity', "parent_atom_indices", "multiplicity_patterns_and_num_hydrogens", "shifts_orig"]}

    return results, overall_error_HSQC, overall_error_COSY



def prepare_nmr_data_for_llm(config: dict, prediction: str, experimental_str: str, nmr_type:str) -> str:
    """
    Prepares NMR data for LLM formatting by combining prediction data and experimental DataFrame.
    
    Args:
        config: Dictionary containing API configuration
        prediction: String containing the prediction data (including structure description and tables)
        experimental_df: DataFrame containing experimental NMR data
        
    Returns:
        Formatted string from LLM containing combined and cleaned data
    """
    # Check if in test mode
    if hasattr(config, 'mode') and config.mode == "Test":
        return """This is a test \n FINISHED \nResults: PASS """

    try:
        # Prepare the request body
        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Please format the following NMR data into a clear, organized report.

                            PREDICTION DATA (string):
                            {prediction}

                            EXPERIMENTAL DATA (raw dataframe):
                            {experimental_str}

                            Please create a well-formatted report that includes:
                            1. A clear description of the molecular structure from the prediction data. Take all of it from the prediction input. don't shorten it. It should be as precise as possible.
                            2. The prediction table of {nmr_type} maintaining its original format
                            3. A new experimental data table formatted like the prediction table
                            4. Ensure all tables are properly aligned and use consistent formatting containing all the information that was presented in the dataframe or prediction string.

                            Note:
                            - If there is an Error column in the dataframe, display it in the same table as the shifts just as a new column. 
                            - Don't invent positions for the experimental data. The shifts are not assigned yet. 

                            Format all tables using the same style as shown in the prediction data, with proper borders and alignment."""
                        }
                    ]
                }
            ]
        }

        # Prepare headers
        headers = {
            "x-api-key": config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        response = make_api_call(client, "claude", request_body=request_body, headers=headers)

        # # Make the API call with the custom headers
        # response = client._client.post(
        #     "https://api.anthropic.com/v1/messages",
        #     json=request_body,
        #     headers=headers
        # )
        response.raise_for_status()
        
        response_data = response.json()
        return response_data['content'][0]['text']

    except Exception as e:
        logging.error(f"Error in prepare_nmr_data_for_llm: {str(e)}")
        import IPython; IPython.embed();
        return response_data['content'][0]['text']

def filter_cosy_unique_pairs(nmr_data: dict) -> dict:
    """
    Filter COSY data to only keep entries where F1 and F2 values are not identical
    with any other pair in the dataset.
    
    Args:
        nmr_data: Dictionary containing 'F1 (ppm)', 'F2 (ppm)', and 'Error' data
    Returns:
        Dictionary with filtered data containing only unique pairs
    """
    # Convert to lists for easier processing
    f1_values = list(nmr_data['F1 (ppm)'].values())
    f2_values = list(nmr_data['F2 (ppm)'].values())
    error_values = list(nmr_data['Error'].values())
    
    # Create a set of all values to check for duplicates
    used_values = set()
    indices_to_keep = []
    
    # Check each pair
    for i in range(len(f1_values)):
        f1 = f1_values[i]
        f2 = f2_values[i]
        
        # If we haven't seen either value before, keep this pair
        if f1 != f2: #in used_values and f2 not in used_values:
            indices_to_keep.append(i)
            #used_values.add(f1)
            #used_values.add(f2)
    
    # Filter the data
    filtered_data = {
        'F1 (ppm)': [f1_values[i] for i in indices_to_keep],
        'F2 (ppm)': [f2_values[i] for i in indices_to_keep],
        'Error': [error_values[i] for i in indices_to_keep]
    }
    
    return filtered_data


def call_agent_o1(config:dict, agent_name: str, nmr_data: Dict[str, Any], question: str, prediction: str) -> str:
    formatted_data = ""
    try:
        # Check if in test mode
        if hasattr(config, 'mode') and config.mode == "Test":
            return """This is a test \n FINISHED \nResults: PASS """

        prompt_file_paths = {
            'hsqc': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/hsqc_system_prompt.txt",
            'cosy': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/cosy_system_prompt.txt",
            '13c': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/13c_system_prompt.txt",
            '1h': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/1h_system_prompt.txt"
        }

        if agent_name not in prompt_file_paths:
            raise ValueError(f"Unknown agent type: {agent_name}")

        # Read the content of the system prompt file
        try:
            with open(prompt_file_paths[agent_name], 'r') as file:
                system_prompt = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The system prompt file for {agent_name} was not found at {prompt_file_paths[agent_name]}")
        except Exception as e:
            raise Exception(f"An error occurred while reading the system prompt file for {agent_name}: {str(e)}")

        # Remove the self_index info and just keep the shift infos and error infos
        #import IPython; IPython.embed();  

        if agent_name in ["hsqc"]:
            exp_data = {
                'F1 (ppm)': list(nmr_data['F1 (ppm)'].values()),
                'F2 (ppm)':list(nmr_data['F2 (ppm)'].values()),
                'Error': list(nmr_data['Error'].values())
            }
        elif agent_name in ["cosy"]:
            exp_data = filter_cosy_unique_pairs(nmr_data)
            #import IPython; IPython.embed();  
        elif agent_name == "1h":
            exp_data = {
                "shifts_orig": {k: v for k, v in nmr_data["shifts_orig"].items() if not pd.isna(v)},
                "multiplicity_patterns_and_num_hydrogens": {
                    k: v for k, v in nmr_data["multiplicity_patterns_and_num_hydrogens"].items() 
                    if v is not None
                }
            }
        elif agent_name == "13c":
            exp_data =  {'shifts': list(nmr_data['shifts'].values())}
        
        formatted_data = prepare_nmr_data_for_llm(config, prediction, json.dumps(exp_data), agent_name)
        
        # Create enhanced prompt with formatted data
        prompt = f"""You are a {agent_name.upper()} NMR analysis agent. {system_prompt}

            Here is the FORMATTED NMR DATA and strucural information of the molecule:
            {formatted_data}

            Request: {question}

            Provide a detailed analysis, referencing the molecular structure, atom labels, and the provided prediction when relevant."""

        log_conversation(config.log_file, f"call_agent_o1", "prompt", prompt)
        #import IPython; IPython.embed(); 

        client = OpenAI(api_key=config.openai_api_key)
        
        print(f"Running o1-mini in call_agent_o1 with {agent_name}")
        response = make_api_call(client, "openai", prompt=prompt)

        # response = client.chat.completions.create(
        #     model="o1-mini",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ]
        # )

        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in prepare_nmr_data_for_llm: {str(e)}")
        import IPython; IPython.embed(); 
        return response.choices[0].message.content

def call_summary_agent_o1(config: dict, specialized_agent_results: Dict[str, str], nmr_data: Dict[str, Any]) -> str:
    try:
        # Check if in test mode
        if hasattr(config, 'mode') and config.mode == "Test":
            return """This is a test \n FINISHED \nResults: PASS """

        prompt_file_path = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/summary_agent_system_prompt.txt"

        # Read the content of the system prompt file
        try:
            with open(prompt_file_path, 'r') as file:
                system_prompt = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The system prompt file was not found at {prompt_file_path}")
        except Exception as e:
            raise Exception(f"An error occurred while reading the system prompt file: {str(e)}")

        # Prepare the prompt for the o1-mini model
        prompt = f"You are a Summary Agent. {system_prompt}\n\nReview the following specialized agent analyses and provide a final conclusion:\n\n{json.dumps(specialized_agent_results, indent=2)}\n\nProvide a detailed analysis and final conclusion, referencing the molecular structure and atom labels when relevant."

        client = OpenAI(api_key=config.openai_api_key)
        
        log_conversation(config.log_file, f"call_summary_agent_o1", "prompt", prompt)

        print(f"Running o1-mini in call_summary_agent_o1 ")
        # Make a request to the o1-mini model
        response = make_api_call(client, "openai", prompt=prompt)
        # response = client.chat.completions.create(
        #     model="o1-mini",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ]
        # )

        # Extract the response text
        response_text = response.choices[0].message.content

        return response_text

    except Exception as e:
        logging.error(f"Error in call_summary_agent_o1: {str(e)}")
        import IPython; IPython.embed();
        return response_text


def call_summary_agent(config:dict, specialized_agent_results: Dict[str, str], image_paths: Dict[str, str], nmr_data: Dict[str, Any]) -> str:
    try:
        # Check if in test mode
        if hasattr(config, 'mode') and config.mode == "Test":
            return """This is a test \n FINISHED \nResults: PASS """

        # Encode images
        image1_data = pl.encode_image(image_paths['normal_view'])
        image2_data = pl.encode_image(image_paths['rotated_view'])
        
        prompt_file_path = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/summary_agent_system_prompt.txt"

        # Read the content of the system prompt file
        try:
            with open(prompt_file_path, 'r') as file:
                system_prompt = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The system prompt file was not found at {prompt_file_path}")
        except Exception as e:
            raise Exception(f"An error occurred while reading the system prompt file: {str(e)}")

        # Prepare the request body
        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2048,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image1_data
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image2_data
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Review the following specialized agent analyses and provide a final conclusion:\n\n{json.dumps(specialized_agent_results, indent=2)}\n\nProvide a detailed analysis and final conclusion, referencing the molecular structure and atom labels when relevant."
                    }
                ]
            }
        ]
        }

        # Prepare headers with API key and beta feature flag
        headers = {
            "x-api-key": config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json"
        }

        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        # Make the API call with the custom headers
        response = make_api_call(client, "claude", request_body=request_body, headers=headers)
        # response = client._client.post(
        #     "https://api.anthropic.com/v1/messages",
        #     json=request_body,
        #     headers=headers
        # )
        # response_data = response.json()

        print("\nUsage statistics for Summary Agent:")
        print(f"Input tokens: {response_data['usage']['input_tokens']}")
        print(f"Output tokens: {response_data['usage']['output_tokens']}")
        print(f"Cache creation input tokens: {response_data['usage'].get('cache_creation_input_tokens', 0)}")
        print(f"Cache read input tokens: {response_data['usage'].get('cache_read_input_tokens', 0)}")
        print("\n" + "-"*50 + "\n")

        return response_data['content'][0]['text'] if response_data['content'][0]['type'] == 'text' else str(response_data['content'])

    except Exception as e:
        logging.error(f"Error in call_summary_agent: {str(e)}")
        import IPython; IPython.embed();
        return response_data['content'][0]['text'] if response_data['content'][0]['type'] == 'text' else str(response_data['content'])

def orchestration_agent(config:dict, guess_smiles: str):
    prompt_file_path = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/system_prompts/orchestration_agent_system_prompt.txt"
    try:
        with open(prompt_file_path, 'r') as file:
            system_prompt = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The system prompt file was not found at {prompt_file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the system prompt file for: {str(e)}")
    # Check if in test mode
    if hasattr(config, 'mode') and config.mode == "Test":
        return """This is a test \n FINISHED \nResults: PASS """

    log_conversation("Orchestration Agent", "system", system_prompt)
    log_conversation("Orchestration Agent", "user", f"Process this SMILES for NMR analysis: {guess_smiles}")

    # Prepare the request body
    request_body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": "You are an NMR analysis orchestration agent."
            },
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Process this SMILES for NMR analysis: {guess_smiles}"
                    }
                ]
            }
        ]
    }

    # Prepare headers with API key and beta feature flag
    headers = {
        "x-api-key": config.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
        "content-type": "application/json"
    }

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    # Make the API call with the custom headers
    response = make_api_call(client, "claude", request_body=request_body, headers=headers)
    # response = client._client.post(
    #     "https://api.anthropic.com/v1/messages",
    #     json=request_body,
    #     headers=headers
    # )

    response_data = response.json()
    response_data = json.loads(response_data)

    log_conversation("Orchestration Agent", "assistant", response_data['content'][0]['text'] if response_data['content'][0]['type'] == 'text' else str(response_data['content']))

    actions = []
    for content in response_data['content']:
        if content['type'] == 'text':
            actions.extend([line.strip() for line in content['text'].split('\n') if line.strip().startswith('ACTION:')])

    print(f"\nOrchestration for SMILES: {guess_smiles}")
    #print(response_data['content'][0]['text'])
    print("\nUsage statistics:")
    print(f"Input tokens: {response_data['usage']['input_tokens']}")
    print(f"Output tokens: {response_data['usage']['output_tokens']}")
    print(f"Cache creation input tokens: {response_data['usage'].get('cache_creation_input_tokens', 0)}")
    print(f"Cache read input tokens: {response_data['usage'].get('cache_read_input_tokens', 0)}")
    print("\n" + "-"*50 + "\n")

    return actions

def delete_inconsistent_shifts(nmr_dict, c13_threshold=3.0, h1_threshold=0.5):
    """
    Delete c13_shift_range or h1_shift_range from the dictionary if they are
    significantly different from the corresponding hsqc_shift_range values.

    :param nmr_dict: Dictionary containing NMR shift ranges
    :param c13_threshold: Threshold for determining significant difference in C13 shifts (default 3.0 ppm)
    :param h1_threshold: Threshold for determining significant difference in H1 shifts (default 0.5 ppm)
    :return: Modified dictionary
    """
    if 'hsqc_shift_range' not in nmr_dict:
        return nmr_dict  # No comparison possible, return original dict

    hsqc_c13, hsqc_h1, _ = nmr_dict['hsqc_shift_range']

    # Check and potentially delete c13_shift_range
    if 'c13_shift_range' in nmr_dict:
        c13_shift, _ = nmr_dict['c13_shift_range']
        if abs(c13_shift - hsqc_c13) > c13_threshold:
            del nmr_dict['c13_shift_range']
            print(f"Deleted c13_shift_range due to inconsistency with hsqc_shift_range")

    # Check and potentially delete h1_shift_range
    if 'h1_shift_range' in nmr_dict:
        h1_shift, _ = nmr_dict['h1_shift_range']
        if abs(h1_shift - hsqc_h1) > h1_threshold:
            del nmr_dict['h1_shift_range']
            print(f"Deleted h1_shift_range due to inconsistency with hsqc_shift_range")

    return nmr_dict


def clear_shift_ranges(nmr_dict):
    """
    Clear hsqc_shift_range, h1_shift_range, and c13_shift_range from the dictionary.

    :param nmr_dict: Dictionary containing NMR shift ranges
    :return: Modified dictionary with specified shift ranges removed
    """
    shift_ranges_to_clear = ['hsqc_shift_range', 'h1_shift_range', 'c13_shift_range']
    
    for shift_range in shift_ranges_to_clear:
        if shift_range in nmr_dict:
            del nmr_dict[shift_range]
            print(f"Deleted {shift_range} from the dictionary")
    
    return nmr_dict


def process_matching_fragments(config, matching_fragments, mol, center_atom_idx, radius, output_path, results_agents, collect_error_results, processed_smi_list):

    # Select the substructure
    submol, amap, submol_connection_points_target, mol_connection_points, connection_mapping_target, _ = select_substructure(mol, center_atom_idx, radius)

    result_smi = ""
    smiles_to_process = []
    
    for idx, matching_fragment in enumerate(matching_fragments):
        try:
            # Get the first matching sample
            sample = matching_fragment['matching_samples'][0]

            connection_mapping_sub = sample["connection_mapping"]

            # Get the replacement molecule
            replacement_mol = sample['submol']
            # Get the connection points
            submol_connection_points = list(sample['connection_mapping'].keys())

            output_file = os.path.join(output_path, f"{idx}.svg")

            # Perform the replacement
            result_mol, saved_image_path, svg = index_preserving_replace_substructure(
                mol, 
                center_atom_idx, 
                radius, 
                replacement_mol, 
                connection_mapping_sub,
                output_file, 
            )

            # Generate and compare NMR data
            result_smi = Chem.MolToSmiles(result_mol)

            if result_smi not in processed_smi_list:
                processed_smi_list.append(result_smi)
                smiles_to_process.append(result_smi)
        except Exception as e:
            print(f"Error processing fragment {matching_fragment['id']}: {str(e)}")

    # Batch process SMILES
    if smiles_to_process:
        list_nmr_data, list_nmr_data_all, list_sdf_paths, list_smi = generate_shifts_batch(config, smiles_to_process)
        #import IPython; IPython.embed();

        for i, (result_smi, nmr_data, nmr_data_all, sdf_path) in enumerate(zip(list_smi, list_nmr_data, list_nmr_data_all, list_sdf_paths)):
            results_agents['guess_nmr_data'] = nmr_data
            results_agents['guess_nmr_data_all'] = nmr_data_all
            results_agents['target_nmr_data'] = results_agents['target_nmr_data']
            results_agents['target_nmr_data_'] = results_agents['target_nmr_data']
            results_agents, overall_error_HSQC, overall_error_COSY = compare_nmr_data(results_agents)
            collect_error_results.append([result_smi, overall_error_HSQC, overall_error_COSY])
            print(f"Processed SMILES {i+1}/{len(smiles_to_process)}: {result_smi}")
    
    return collect_error_results, processed_smi_list



def find_longer_smiles_index(data_list):
    # for finding the longest smiles from the revisit list of dict to take as a reference
    # This is for the case if we use radius greater than 1
    smiles_list = [item[2][1]['smiles'] for item in data_list]
    longest_smiles_index = max(range(len(smiles_list)), key=lambda i: len(smiles_list[i]))
    return longest_smiles_index




def canonicalize_smiles_list(smiles_list: List[str], 
                            remove_hydrogens: bool = True,
                            remove_atom_mapping: bool = True) -> List[Optional[str]]:
    """
    Canonicalize a list of SMILES strings using RDKit.
    
    Parameters:
    -----------
    smiles_list : List[str]
        List of SMILES strings to canonicalize
    remove_hydrogens : bool, optional (default=True)
        Whether to remove explicit hydrogens in the output
    remove_atom_mapping : bool, optional (default=True)
        Whether to remove atom mapping numbers
        
    Returns:
    --------
    List[Optional[str]]
        List of canonicalized SMILES strings. Invalid SMILES return None.
    """
    canonical_smiles = []
    
    for smiles in smiles_list:
        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                continue
            
            # Remove atom mapping if requested
            if remove_atom_mapping:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
            
            # Generate canonical SMILES
            if remove_hydrogens:
                mol = Chem.RemoveHs(mol)
            else:
                mol = Chem.AddHs(mol)
                
            canonical_smiles.append(Chem.MolToSmiles(mol, 
                                                   canonical=True,
                                                   isomericSmiles=True))
            
        except Exception as e:
            continue
    
    return canonical_smiles

"""
def execute_actions(guess_smiles, actions, target_nmr_data, master_KG, agent_types, compare_only=False):

    output_path = "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/trash_folder"
    results_agents = {}
    sdf_path = None
    nmr_data = None
    image_paths = None
    specialized_agent_results = None
    knowledge_graph = None
    incorrect_part = None
    corrected_kg = None
    substitution_pattern = None
    gen_sdf_path = None


    # import IPython; IPython.embed();

    while actions:
        current_action = actions.pop(0)

        if 'generate_shifts' in current_action:
            print("generate_shifts")
            guess_nmr_data, guess_nmr_data_all, gen_sdf_path, list_smi = generate_shifts_batch(config, guess_smiles)
            #import IPython; IPython.embed();
            results_agents['guess_nmr_data'] = guess_nmr_data[0]
            results_agents['guess_nmr_data_all'] = guess_nmr_data_all[0]
            results_agents['target_nmr_data'] = target_nmr_data
            results_agents['target_nmr_data_'] = target_nmr_data

        elif 'compare_nmr_data' in current_action:
            results_agents, overall_error_HSQC, overall_error_COSY = compare_nmr_data(results_agents)

        elif 'generate_molecule_images' in current_action:
            print("generate_molecule_images")
            image_paths = pl.generate_molecule_images(guess_smiles, output_path)
            results_agents['image_paths'] = image_paths
            results_agents["guess_smiles"] = guess_smiles

        elif 'send_to_specialized_agents' in current_action:
            print("send_to_specialized_agents")
            specialized_agent_results = send_to_specialized_agents(results_agents['target_nmr_data_'], results_agents['image_paths'], agent_types)
            results_agents['specialized_agent_results'] = specialized_agent_results
            log_conversation("Specialized Agent", "assistant", specialized_agent_results)

        elif 'call_summary_agent' in current_action:
            print("call_summary_agent")
            summary =None
            while type(summary) == type(None):
                summary = call_summary_agent(results_agents['specialized_agent_results'], results_agents['image_paths'], results_agents['target_nmr_data_'])

            results_agents['summary'] = summary
            revisit_numbers, remove_numbers, quality_score = extract_information(results_agents["summary"])
            log_conversation("Summary Agent", "assistant", summary)
            if compare_only or quality_score > 0.799:
                print(results_agents, gen_sdf_path)
                return results_agents, gen_sdf_path

                if not compare_only:
                    if len(remove_numbers) != 0:
                        for atom_index in remove_numbers:
                            actions.insert(0, f'remove_atom_by_index({atom_index})')    
                    elif len(revisit_numbers) != 0:
                        for atom_index in revisit_numbers:
                            actions.insert(0, f'create_knowledge_graph') 

        elif 'remove_atom_by_index' in current_action:
            print("remove_atom_by_index")
            atom_index = int(current_action.split('(')[1].split(')')[0])

            mol = Chem.MolFromSmiles(guess_smiles)
            if is_end_group(mol, atom_index):
                new_smiles = remove_atom_by_index(guess_smiles, atom_index)
                if new_smiles:
                    guess_smiles = new_smiles
                    actions.insert(0, 'call_summary_agent()')  # Re-run generate_shifts with new SMILES
                    actions.insert(0, 'send_to_specialized_agents()')  # Re-run generate_shifts with new SMILES
                    actions.insert(0, 'generate_molecule_images()')  # Re-run generate_shifts with new SMILES
                    actions.insert(0, 'compare_nmr_data()')  # Re-run generate_shifts with new SMILES
                    actions.insert(0, 'generate_shifts()')  # Re-run generate_shifts with new SMILES
                    print("remove atom by index")                   
                    import IPython; IPython.embed();                   
            else:
                print(f"Atom at index {index} is not an end-group and cannot be safely removed.")

        elif 'create_knowledge_graph' in current_action:
            print("create_knowledge_graph")
            #import IPython; IPython.embed();     
            dict_hsqc = results_agents["guess_nmr_data_all"]["HSQC"]
            dict_cosy = results_agents["guess_nmr_data_all"]["COSY"]
            dict_1h = results_agents["guess_nmr_data_all"]["1H"]
            dict_13C = results_agents["guess_nmr_data_all"]["13C"]
            # Assuming dict_13C is a dictionary
            df_13c = pd.DataFrame(dict_13C)
            df_1h = pd.DataFrame(dict_1h)
            df_hsqc = pd.DataFrame(dict_hsqc)
            df_cosy = pd.DataFrame(dict_cosy)
            combined_data, example_molecule = ed.combine_nmr_data(gen_sdf_path[0], df_13c, df_1h, df_hsqc, df_cosy)
            G, graph_data = kg.create_knowledge_graph([example_molecule[0]], combined_data, radius)
            #revisit_filtered_fragment = filter_has_fragment(graph_data, revisit_numbers)
            revisit_filtered_fragment = filter_has_fragment(graph_data, revisit_numbers[0])

            if radius >1:
                index_longest = find_longer_smiles_index(revisit_filtered_fragment)
                revisit_filtered_fragment = [revisit_filtered_fragment[index_longest]]
            mol = Chem.SDMolSupplier(gen_sdf_path[0], removeHs=True)[0]
            #import IPython; IPython.embed();             

            #work out a logic that gets the new shifts from the real data into the selected graph of the substructure
            actions.insert(0, 'search_in_knowledge_graph()')  # Re-run generate_shifts with new SMILES


        elif 'search_in_knowledge_graph' in current_action:
            print("search_filter_knowledge_graph")
            filters = prepare_search_filters(summary, revisit_filtered_fragment, weight_delta=15, hsqc_delta=10, c13_delta=10, h1_delta=10)
            filters = delete_inconsistent_shifts(filters)
            matching_fragments = kg.filter_fragments(master_KG, filters)      
            center_atom_idx = revisit_numbers[0]
            actions.insert(0, 'index_preserving_replace_substructure()')
            #import IPython; IPython.embed();             

        # Usage in the main code:
        elif 'index_preserving_replace_substructure' in current_action:
            print("index_preserving_replace_substructure")
            collect_error_results = []
            processed_smi_list = []
            collect_error_results, processed_smi_list = process_matching_fragments(config, matching_fragments, mol, center_atom_idx, radius, output_path, results_agents, collect_error_results, processed_smi_list)
            
            ### Run it also with previously guessed molecule
            #guess_nmr_data, guess_nmr_data_all, gen_sdf_path = generate_shifts_batch(processed_smi_list)
            #results_agents['guess_nmr_data'] = guess_nmr_data[0]
            #results_agents['guess_nmr_data_all'] = guess_nmr_data_all[0]
            #results_agents['target_nmr_data'] = target_nmr_data[0]
            #results_agents['target_nmr_data_'] = target_nmr_data[0]
            #results_agents, overall_error_HSQC, overall_error_COSY = compare_nmr_data(results_agents)
            #revisit_filtered_fragment = filter_has_fragment(graph_data, revisit_numbers[0])
            
            ### Add the previously guessed molecule to the list 
            collect_error_results.append([guess_smiles, overall_error_HSQC, overall_error_COSY])            
            
            # Sort results and select the best one
            sorted_results = sorted(collect_error_results, key=lambda x: x[1])
            guess_smiles_new = sorted_results[0][0] if sorted_results else None                               
            
            if guess_smiles_new == guess_smiles or collect_error_results == []:
                if collect_error_results != []:
                    collect_error_results = sorted_results[1:]
                print("without NMR Filter")
                filters = clear_shift_ranges(filters)
                matching_fragments = kg.filter_fragments(master_KG, filters)                      
                collect_error_results, processed_smi_list = process_matching_fragments(config, matching_fragments, mol, center_atom_idx, radius, output_path, results_agents, collect_error_results, processed_smi_list)

                ### Run it also with previously guessed molecule
                #results_agents['guess_nmr_data'] = guess_nmr_data[0]
                #results_agents['guess_nmr_data_all'] = guess_nmr_data_all[0]
                #results_agents['target_nmr_data'] = target_nmr_data[0]
                #results_agents['target_nmr_data_'] = target_nmr_data[0]
                #results_agents, overall_error_HSQC, overall_error_COSY = compare_nmr_data(results_agents)
                #revisit_filtered_fragment = filter_has_fragment(graph_data, revisit_numbers[0])
                collect_error_results.append([guess_smiles, overall_error_HSQC, overall_error_COSY])            
            
                # Sort results and select the best one
                sorted_results = sorted(collect_error_results, key=lambda x: x[1])
                guess_smiles_new = sorted_results[0][0] if sorted_results else None                               
            
                if guess_smiles_new == guess_smiles or collect_error_results == []:
                    if collect_error_results != []:
                        collect_error_results = sorted_results[1:]
                    print("without weight Filter")
                    try:
                        del filters["weight_range"]
                    except:
                        pass
                    matching_fragments = kg.filter_fragments(master_KG, filters)
                    collect_error_results, processed_smi_list = process_matching_fragments(config, matching_fragments, mol, center_atom_idx, radius, output_path, results_agents, collect_error_results, processed_smi_list)
                    
                    ### Run it also with previously guessed molecule
                    #results_agents['guess_nmr_data'] = guess_nmr_data
                    #results_agents['guess_nmr_data_all'] = guess_nmr_data_all
                    #results_agents['target_nmr_data'] = target_nmr_data
                    #results_agents['target_nmr_data_'] = target_nmr_data
                    #results_agents, overall_error_HSQC, overall_error_COSY = compare_nmr_data(results_agents)
                    #revisit_filtered_fragment = filter_has_fragment(graph_data, revisit_numbers[0])
                    collect_error_results.append([guess_smiles, overall_error_HSQC, overall_error_COSY])            

                    # Sort results and select the best one
                    sorted_results = sorted(collect_error_results, key=lambda x: x[1])
                    guess_smiles_new = sorted_results[0][0] if sorted_results else None                               
            
            if guess_smiles_new == guess_smiles:
                print("repeat analysis")
            else:
                guess_smiles = guess_smiles_new                

            # Add next actions to the queue
            actions.insert(0, 'send_to_specialized_agents()')
            actions.insert(0, 'call_summary_agent()')
            actions.insert(0, 'send_to_specialized_agents()')
            actions.insert(0, 'generate_molecule_images()')
            actions.insert(0, 'compare_nmr_data()')
            actions.insert(0, 'generate_shifts()')
            print("Finished index_preserving_replace_substructure")
            import IPython; IPython.embed();
            #### until all structures are finished compared to input

    return results_agents, gen_sdf_path
#### REPLACE FUNCTION with code below
"""
