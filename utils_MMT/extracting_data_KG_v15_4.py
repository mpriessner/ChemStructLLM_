


# Standard library imports
import os
import collections
import base64
import io
import re
import math
import random
import csv
from datetime import datetime


# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, SVG
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# RDKit imports
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Draw, SDMolSupplier, MolToSmiles, AddHs, Descriptors,
    rdMolDescriptors, rdDepictor, rdMolTransforms, rdmolfiles, PandasTools
)
from rdkit.Chem.Draw import IPythonConsole, MolToImage, rdMolDraw2D
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

import utils_MMT.nmr_calculation_from_dft_v15_4 as ncfd


############################################################################################
####################################### HSQC ###############################################
############################################################################################



def get_molecule_data(compound_path):
    atom_list = []
    connectivity_list = []
    docline_list = []
    with open(compound_path) as f:
        docline_list = f.readlines()
    
    start = False
    counter = 0
    for i in docline_list:
        if start:
            if i.split()[0].isdigit():
                break
            atom = i.split()[3]
            if atom != "0":
                atom_list.append(atom)
                counter += 1
        if "V2000" in i:
            start = True
    
    start_line = len(atom_list) + 4
    for idx, i in enumerate(docline_list):
        if idx >= start_line and "M  END" not in i:
            add_list = i.split()
            if add_list[0] not in ["M", ">", "$$$$"]:
                connectivity_list.append(add_list)
        if "M  END" in i:
            break
    
    mol = Chem.MolFromMolFile(compound_path[:-4] + ".sdf")
    return atom_list, connectivity_list, docline_list, mol



def mol_with_atom_index(mol, include_H=True):
    if include_H:
        mol = Chem.AddHs(mol, addCoords=True)
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

"""
def run_HSQC_generation(dft_file_path):

    boltzman_avg_shifts_corr_2 = ncfd.load_shifts_from_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = ncfd.run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol  = ncfd.get_molecule_data(dft_file_path)
    c_h_connectivity_dict = ncfd.get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = ncfd.selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = ncfd.perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_hsqc = ncfd.generate_dft_dataframe(shifts)

    # Generate canonicalized SMILES
    mol = Chem.MolFromMolFile(dft_file_path)
    isomers = tuple(EnumerateStereoisomers(mol))
    canonicalized_smiles = Chem.MolToSmiles(isomers[0],  isomericSmiles=False, canonical=True)  

    # Extract sample-id from file path
    file_name = os.path.basename(dft_file_path)
    sample_id = file_name.split('NMR_')[-1].split('.')[0]
    
    # Add SMILES and sample-id columns to the dataframe
    df_hsqc['SMILES'] = canonicalized_smiles
    df_hsqc['sample-id'] = sample_id
    
    return df_hsqc, canonicalized_smiles"""


def run_HSQC_generation(dft_file_path):
    # Load shifts and perform necessary calculations
    boltzman_avg_shifts_corr_2 = ncfd.load_shifts_from_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = ncfd.run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = ncfd.get_molecule_data(dft_file_path)
    c_h_connectivity_dict = ncfd.get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = ncfd.selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = ncfd.perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_hsqc = ncfd.generate_dft_dataframe(shifts)

    # Generate canonicalized SMILES
    mol = Chem.MolFromMolFile(dft_file_path)
    isomers = tuple(EnumerateStereoisomers(mol))
    canonicalized_smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=False, canonical=True)  

    # Extract sample-id from file path
    file_name = os.path.basename(dft_file_path)
    sample_id = file_name.split('NMR_')[-1].split('.')[0]
    
    # Add SMILES and sample-id columns to the dataframe
    df_hsqc['SMILES'] = canonicalized_smiles
    df_hsqc['sample-id'] = sample_id
    
    # Keep a copy of the combined dataframe (before splitting)
    df_hsqc_combined = df_hsqc.copy()

    # Split rows where atom_index has multiple numbers (e.g., '1_2_3')
    #df_hsqc['atom_index'] = df_hsqc['atom_index'].str.split('_')
    df_hsqc['atom_index'] = df_hsqc['atom_index'].str.split('_').apply(lambda x: list(map(int, x)))
    
    df_hsqc_expanded = df_hsqc.explode('atom_index').reset_index(drop=True)

    # Return both DataFrames and the SMILES
    return df_hsqc_combined, df_hsqc_expanded, canonicalized_smiles



############################################################################################
####################################### COSY ###############################################
############################################################################################

def find_chiral_centers(molecule):
    return [atom.GetIdx() for atom in molecule.GetAtoms() 
            if atom.GetAtomicNum() == 6 and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]

def find_carbons_with_relevant_neighbors(molecule):
    carbon_dict = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0:
            neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() 
                         if neighbor.GetAtomicNum() == 6 and neighbor.GetTotalNumHs() > 0]
            carbon_dict[atom.GetIdx()] = neighbors
    return carbon_dict

def find_heavy_atoms_with_hydrogens(molecule):
    return {atom.GetIdx(): atom.GetTotalNumHs() for atom in molecule.GetAtoms() 
            if atom.GetAtomicNum() != 1 and atom.GetTotalNumHs() > 0}

def extract_symmetric_hydrogen_shifts(shifts, heavy_atom_dict):
    num_heavy_atoms = len(heavy_atom_dict)
    hydrogen_shifts = shifts[num_heavy_atoms:]
    carbon_hydrogen_shifts_dict = {}
    for carbon, num_hydrogens in sorted(heavy_atom_dict.items(), key=lambda x: x[0], reverse=True):
        carbon_hydrogen_shifts_dict[carbon] = hydrogen_shifts[-num_hydrogens:]
        hydrogen_shifts = hydrogen_shifts[:-num_hydrogens]
    return carbon_hydrogen_shifts_dict

def find_symmetric_positions(stereo_smi):
    mol = Chem.MolFromSmiles(stereo_smi)
    z = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    matches = mol.GetSubstructMatches(mol, uniquify=False)
    if len(z) != len(set(z)) and len(matches) > 1:
        symmetric = [item for item, count in collections.Counter(z).items() if count > 1]
        example_match = matches[0]
        return [[example_match[i] for i, v in enumerate(z) if v == j] for j in symmetric]
    return []

def has_hydrogens(mol, atom_idx):
    return mol.GetAtomWithIdx(atom_idx).GetTotalNumHs() > 0

def average_shifts(shift_list, sym_groups):
    return {i: [sum([shift_list.get(j, [0])[0] for j in group]) / len(group)]
            for group in sym_groups for i in group}

def update_shifts_with_averaged(original_shifts, averaged_shifts):
    updated_shifts = original_shifts.copy()
    updated_shifts.update(averaged_shifts)
    return updated_shifts

def plot_and_save_cosy_spectrum_with_zoom_no_duplicates(heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers):
    plotted_points = set()
    for carbon1, neighbors in carbon_dict.items():
        h1_shifts = heavy_atom_hydrogen_shift_dict.get(carbon1, [])
        if h1_shifts:
            plotted_points.add((h1_shifts[0], h1_shifts[0], carbon1, carbon1))
        for carbon2 in neighbors:
            h2_shifts = heavy_atom_hydrogen_shift_dict.get(carbon2, [])
            if h1_shifts and h2_shifts:
                is_chiral = carbon1 in chiral_centers or carbon2 in chiral_centers
                if is_chiral:
                    for h1_shift in h1_shifts:
                        for h2_shift in h2_shifts:
                            plotted_points.add((h1_shift, h2_shift, carbon1, carbon2))
                            plotted_points.add((h2_shift, h1_shift, carbon2, carbon1))
                else:
                    plotted_points.add((h1_shifts[0], h2_shifts[0], carbon1, carbon2))
                    plotted_points.add((h2_shifts[0], h1_shifts[0], carbon2, carbon1))
    return list(plotted_points)

def run_COSY_generation(sdf_file_path):
    try:
        mol = SDMolSupplier(sdf_file_path)[0]
        isomers = tuple(EnumerateStereoisomers(mol))
        stereo_smi = Chem.MolToSmiles(isomers[0],  isomericSmiles=False, canonical=True)  
        averaged_nmr_shifts = mol.GetProp('averaged_NMR_shifts')
        sample_shifts = list(map(float, averaged_nmr_shifts.split()))
        sample_id = os.path.splitext(os.path.basename(sdf_file_path))[0].split('NMR_')[-1]
        
        chiral_centers = find_chiral_centers(mol)
        carbon_dict = find_carbons_with_relevant_neighbors(mol)
        heavy_atom_dict = find_heavy_atoms_with_hydrogens(mol)
        heavy_atom_hydrogen_shift_dict = extract_symmetric_hydrogen_shifts(sample_shifts, heavy_atom_dict)
        sym_dupl_lists = find_symmetric_positions(stereo_smi)
        sym_dupl_lists = [positions for positions in sym_dupl_lists if all(has_hydrogens(mol, idx) for idx in positions)]
        averaged_shifts = average_shifts(heavy_atom_hydrogen_shift_dict, sym_dupl_lists)
        updated_heavy_atom_hydrogen_shift_dict = update_shifts_with_averaged(heavy_atom_hydrogen_shift_dict, averaged_shifts)
        
        COSY_shifts = plot_and_save_cosy_spectrum_with_zoom_no_duplicates(updated_heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers)
        
        df_cosy = pd.DataFrame(COSY_shifts, columns=['F2 (ppm)', 'F1 (ppm)', 'atom_index_1', 'atom_index_2'])
        df_cosy['atom_index'] = df_cosy['atom_index_1'].astype(str) + '_' + df_cosy['atom_index_2'].astype(str)

        df_cosy['SMILES'] = MolToSmiles(Chem.RemoveHs(mol))
        df_cosy['sample-id'] = sample_id
        
        return df_cosy, stereo_smi
    except Exception as e:
        print(f"Error processing file {sdf_file_path}: {str(e)}")
        return None


############################################################################################
######################################## 13C ###############################################
############################################################################################



def find_symmetric_positions(stereo_smi):
    mol = Chem.MolFromSmiles(stereo_smi)
    z = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    matches = mol.GetSubstructMatches(mol, uniquify=False)
    
    if len(z) != len(set(z)) and len(matches) > 1:
        symmetric = [item for item, count in collections.Counter(z).items() if count > 1]
        example_match = matches[0]
        sym_dupl_lists = []
        for j in symmetric:
            indices = [example_match[i] for i, v in enumerate(z) if v == j]
            sym_dupl_lists.append(indices)
        return sym_dupl_lists
    return []

def consolidate_peaks(averaged_shifts, symmetric_positions):
    consolidated_shifts = averaged_shifts.copy()
    for positions in symmetric_positions:
        avg_value = sum(averaged_shifts[i] for i in positions) / len(positions)
        for i in positions:
            consolidated_shifts[i] = avg_value
    return consolidated_shifts



def run_13C_generation(sdf_file_path):
    try:
        # Load molecule from SDF file
        mol = SDMolSupplier(sdf_file_path)[0]
        isomers = tuple(EnumerateStereoisomers(mol))
        stereo_smi = Chem.MolToSmiles(isomers[0],  isomericSmiles=False, canonical=True)  

        # Get the NMR shifts
        averaged_nmr_shifts = mol.GetProp('averaged_NMR_shifts')
        sample_shifts = list(map(float, averaged_nmr_shifts.split()))

        # Extract the sample_id from the path
        file_name = os.path.basename(sdf_file_path)
        sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

        # Remove symmetric carbons
        sym_dupl_lists = find_symmetric_positions(stereo_smi)
        sym_corr_nmr_shifts = consolidate_peaks(sample_shifts, sym_dupl_lists)

        # Get carbon atoms and their indices
        carbon_atoms = [(atom.GetIdx(), shift) for atom, shift in zip(mol.GetAtoms(), sym_corr_nmr_shifts) 
                        if atom.GetSymbol() == 'C' and shift != 0]

        # Sort by shift value
        carbon_atoms.sort(key=lambda x: x[1])

        # Create original DataFrame
        df_13c_separated = pd.DataFrame(carbon_atoms, columns=['atom_index', 'shifts'])

        # Add SMILES and sample-id columns
        mol = Chem.RemoveHs(mol)
        smi = MolToSmiles(mol)
        df_13c_separated['SMILES'] = smi
        df_13c_separated['sample-id'] = sample_id

        # Create consolidated DataFrame
        df_13c = df_13c_separated.groupby('shifts').agg({
            'atom_index': lambda x: '_'.join(map(str, sorted(x))),
            'SMILES': 'first',
            'sample-id': 'first'
        }).reset_index()

        return df_13c, df_13c_separated

    except Exception as e:
        print(f"Error processing file {sdf_file_path}: {str(e)}")
        return None, None



############################################################################################
######################################## 1H ################################################
############################################################################################


# Add this function to read shifts from the SDF file
def read_shifts_from_sdf(file_path):
    supplier = SDMolSupplier(file_path)
    sdf_mol = supplier[0]  # assuming there is only one molecule in the file
    shifts = {}
    for atom in sdf_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_shift = atom.GetProp("_Shift")
        shifts[atom_idx] = float(atom_shift)
    return shifts

def lorentzian(x, x0, gamma):
    return (1 / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)


def simulate_splitting(shifts, coupling_patterns, gamma, spectrometer_frequency):
    x = np.linspace(shifts.min() - 1, shifts.max() + 1, 1000)
    y = np.zeros_like(x)
    for shift, coupling_pattern in zip(shifts, coupling_patterns):
        peak = np.zeros_like(x)
        for J, intensity in coupling_pattern:
            peak += intensity * lorentzian(x, shift + J / spectrometer_frequency, gamma)
        y += peak
    return x, y

def get_adjacent_aromatic_hydrogens(atom):
    aromatic_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetIsAromatic()]
    aromatic_hydrogens = []
    for aromatic_neighbor in aromatic_neighbors:
        aromatic_hydrogens.extend(get_surrounding_hydrogens(aromatic_neighbor))
    return aromatic_hydrogens

def get_surrounding_hydrogens(atom):
    neighboring_hydrogens = []
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'H':
            neighboring_hydrogens.append((neighbor, neighbor.GetIdx()))
    return neighboring_hydrogens

def analyze_molecule(mol):
    hydrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    nmr_data = []
    assigned_shifts = {}

    for hydrogen in hydrogens:
        parent_atom = hydrogen.GetNeighbors()[0]
        is_aromatic = parent_atom.GetIsAromatic()
        group_key = (parent_atom.GetSymbol(), parent_atom.GetIdx())

        surrounding_hydrogens = get_surrounding_hydrogens(parent_atom)
        num_h_neighbors = len(surrounding_hydrogens) - 1

        hydrogen_label = f"{parent_atom.GetSymbol()}{parent_atom.GetIdx()}H{num_h_neighbors + 1 - surrounding_hydrogens.count(hydrogen)}"

        nmr_data.append({
            'atom': hydrogen,
            'atom_index': hydrogen.GetIdx(),
            'parent_atom_index': parent_atom.GetIdx(),  # Add this line
            'aromatic': is_aromatic,
            'neighbors': num_h_neighbors,
            'label': hydrogen_label,
            'group_key': group_key
        })

        assigned_shifts[group_key] = float(hydrogen.GetProp("_Shift"))
    
    return nmr_data, assigned_shifts, mol


def pascals_triangle(n):
    if n == 0:
        return [1]
    else:
        previous_row = pascals_triangle(n - 1)
        current_row = [1]
        for i in range(len(previous_row) - 1):
            current_row.append(previous_row[i] + previous_row[i + 1])
        current_row.append(1)
        return current_row

def generate_nmr_coupling_pattern(n_neighbors, J):
    coefficients = pascals_triangle(n_neighbors)
    intensities = [coef / (2 ** n_neighbors) for coef in coefficients]
    Js = [i * J for i in range(-n_neighbors // 2, n_neighbors // 2 + 1)]
    return list(zip(Js, intensities))


def load_mol_and_assign_shifts(file_path):
    data = PandasTools.LoadSDF(file_path)
    mol = data["ROMol"].item()
    mol = AddHs(mol, addCoords=True)

    str_shifts = data["averaged_NMR_shifts"].item()
    shifts  = [float(i) for i in str_shifts.split()]

    atoms = list(mol.GetAtoms())
    i = 0
    for idx, atom in enumerate(atoms):
        atom.SetProp("_Shift", str(shifts[idx]))
    mol = AddHs(mol, addCoords=False)

    return mol

def add_shifts_to_data(nmr_data, assigned_shifts):
    grouped_shifts = {}
    for atom_data in nmr_data:
        group_key = atom_data['group_key']
        if group_key not in grouped_shifts:
            grouped_shifts[group_key] = []
        if group_key in assigned_shifts:
            grouped_shifts[group_key].append((assigned_shifts[group_key], atom_data['atom_index'], atom_data['parent_atom_index']))  # Include both indices

    avg_shifts = {group_key: np.mean([shift for shift, _, _ in shifts if np.isfinite(shift)]) for group_key, shifts in grouped_shifts.items()}

    for atom_data in nmr_data:
        group_key = atom_data['group_key']
        if group_key in avg_shifts:
            atom_data['shift'] = avg_shifts[group_key]
    return nmr_data

def calculate_couplings_constants(nmr_data):
    
    J_aromatic = 8.0

    ### Version 2
    # Calculate coupling patterns using the average shifts
    atoms_done = []
    coupling_patterns = []
    hydrogen_num = []
    shifts = []
    hydrogen_counts = None
    atom_indices = []
    parent_atom_indices = []  # Add this line

    for atom_data in nmr_data:
        #import IPython; IPython.embed();
        
        if ("N" in atom_data["label"] or "O" in atom_data["label"]) :
            continue
        parent_atom = atom_data['atom'].GetNeighbors()[0]
        if (atom_data['aromatic'] and atom_data['label'] not in atoms_done):
            n_neighbors = atom_data['neighbors']
            adjacent_aromatic_hydrogens = get_adjacent_aromatic_hydrogens(parent_atom)
            arom_n_neighbors = len(adjacent_aromatic_hydrogens)
            if arom_n_neighbors == 0:
                coupling_patterns.append([(J_aromatic, 1)])
            else:
                coupling_patterns.append(generate_nmr_coupling_pattern(arom_n_neighbors, J_aromatic))
            shifts.append(atom_data['shift'])
            atoms_done.append(atom_data['label'])
            hydrogen_num.append(atom_data['neighbors']+1)
            atom_indices.append(atom_data['atom_index'])
            parent_atom_indices.append(atom_data['parent_atom_index'])  # Add this line
            
        elif atom_data['label'] not in atoms_done:

            bond_types = [bond.GetBondType() for bond in parent_atom.GetBonds() if bond.GetOtherAtom(parent_atom).GetSymbol() == 'C']        

            n_neighbors = atom_data['neighbors']

            carbon_neighbors = [neighbor for neighbor in parent_atom.GetNeighbors() if neighbor.GetSymbol() == 'C']

            hydrogen_counts = [sum(1 for neighbor in carbon_neighbor.GetNeighbors() if neighbor.GetSymbol() == 'H') for carbon_neighbor in carbon_neighbors]
            #print(hydrogen_counts, atom_data["label"])
            # Rule-based coupling pattern generation
            if hydrogen_counts == [] and n_neighbors == 2:
                #N-CH3
                coupling_pattern = [(0, 3)]   

            if hydrogen_counts == [] and n_neighbors == 1:
                #CCl2=CH2
                coupling_pattern = [(0, 2)]  
                
            if hydrogen_counts == [] and n_neighbors == 0:
                #(CCl2)3-CH
                coupling_pattern = [(0, 0)]  

            if hydrogen_counts == [0] and n_neighbors == 2:
                coupling_pattern = [(0, 3)]

            if hydrogen_counts == [0] and n_neighbors == 1:
                coupling_pattern = [(0, 2)]   

            if hydrogen_counts == [0] and n_neighbors == 0:
                coupling_pattern = [(0, 1)]  


            if hydrogen_counts == [0, 0] and n_neighbors == 0:
                coupling_pattern = [(0, 1)]  
            
            if hydrogen_counts == [0, 0] and n_neighbors == 1:
                coupling_pattern = [(0, 2)]   

            if hydrogen_counts == [1] and Chem.rdchem.BondType.DOUBLE in bond_types and n_neighbors == 1:
                # CH=CH2 case J = 16 10
                ### Approximation
                J_doublet_1 = 16  
                J_doublet_2 = 10 
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/2), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/2),
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/2), 
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/2)]

            elif hydrogen_counts == [1] and Chem.rdchem.BondType.SINGLE in bond_types and n_neighbors == 1:
                # CH-CH2-Cl case J=5.9
                J_doublet = 5.9  # Coupling constant for the single bond between the CH hydrogens
                coupling_pattern = [(-0.5*J_doublet, 1), 
                                      (0.5*J_doublet, 1)]

            elif hydrogen_counts == [1] and Chem.rdchem.BondType.SINGLE in bond_types and n_neighbors == 2:
                # CH-CH3 case J = 6.1
                J_doublet = 6.1  # Coupling constant for the single bond between the CH hydrogens
                coupling_pattern = [(-0.5*J_doublet, 1.5), 
                                      (0.5*J_doublet, 1.5)]
                
            if hydrogen_counts == [1] and n_neighbors == 0:
                #(CCl2)2-CH-CHCl2                
                J_doublet = 6.1
                coupling_pattern = [(-0.5*J_doublet, 0.5),
                                    (0.5*J_doublet, 0.5)]  
                
            #elif hydrogen_counts == [2] and Chem.rdchem.BondType.DOUBLE in bond_types:
            #    # CH2=CH2 case
            #    J_triplet = J_double_bond  # Coupling constant for the double bond between the CH2=CH2 hydrogens
            #    coupling_pattern = [(-J_triplet, 1/2), 
            #                          (0, 2/2), 
            #                          (J_triplet, 1/2)]

            elif hydrogen_counts == [2] and Chem.rdchem.BondType.SINGLE in bond_types:
                # CH2-CH2 case J = 6.3
                J_triplet = 6.3  # Coupling constant for the single bond between the CH2-CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]

            elif hydrogen_counts == [3] and Chem.rdchem.BondType.SINGLE in bond_types:
                # CH3-CH2 case J = 7
                J_quartet =   7.0
                coupling_pattern = [(-1.5*J_quartet, 2/6), 
                                      (-0.5*J_quartet, 4/6), 
                                      (0.5*J_quartet, 4/6), 
                                      (1.5*J_quartet, 2/6)]

            elif (hydrogen_counts == [1,0] or hydrogen_counts == [0,1]):
                # CH-CH2-CO case J = 7
                J_doublet = 6.9  
                coupling_pattern = [(-0.5*J_doublet, 1), 
                                      (0.5*J_doublet, 1)]            


            elif (hydrogen_counts == [2,0] or hydrogen_counts == [0,2]) and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH2=CH-CO case J = 7
                J_doublet_1 = 18  # Coupling constant for the double bond between the CH=CH2 hydrogens
                J_doublet_2 = 10  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/4),
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/4)]  

            elif (hydrogen_counts == [2,0] or hydrogen_counts == [0,2]) and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH2-CH2-CO case J = 7
                J_triplet = 6.7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]   

            elif hydrogen_counts == [3,0]  or hydrogen_counts == [0,3]:
                # CH3-CHCl-CO case J = 7
                J_quartet = 7  
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]
                
            elif hydrogen_counts == [1, 1] and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH-CH=CH non-aromatic case  13/6.06
                J_doublet_1 = 6.06  # Coupling constant for the single bond between CH hydrogens
                J_doublet_2 = 13 # Coupling constant for the double bond between the CH=CH hydrogens
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/4),
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/4)]

            elif hydrogen_counts == [1, 1] and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH-CH2-CH non-aromatic case  13/6.06
                J_triplet = 6.0  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]   

            elif (hydrogen_counts == [1, 2] or hydrogen_counts == [2, 1]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 1 and bond_types.count(Chem.rdchem.BondType.DOUBLE) == 1:  
                # CH=CH-CH2 case  J = 7.4 
                J_quartet = 7.4  # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]

            elif (hydrogen_counts == [1, 2] or hydrogen_counts == [2, 1]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH-CH2-CH2 case  J = 7.4 
                # an approximation
                J_quartet = 7.4   # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 2/6), 
                                      (-0.5*J_quartet, 4/6), 
                                      (0.5*J_quartet, 4/6), 
                                      (1.5*J_quartet, 2/6)]

            elif hydrogen_counts == [2, 2] and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH2-CH2-CH2 case (quintet)  J=6.57
                J_quintet = 6.57
                coupling_pattern = [(-2 * J_quintet, 2/9), 
                                    (-J_quintet, 4/9), 
                                    (0, 6/9), 
                                    (J_quintet, 4/9), 
                                    (2 * J_quintet, 2/9)]

            elif (hydrogen_counts == [3, 1] or hydrogen_counts == [1, 3]) and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH3-CH=CH case or CH=CH-CH3 J = 7 (Douplet of quartet)
                # an approximation
                J_quintet = 7
                coupling_pattern = [(-2 * J_quintet, 1/9), 
                                    (-J_quintet, 2/9), 
                                    (0, 3/9), 
                                    (J_quintet, 2/9), 
                                    (2 * J_quintet, 1/9)]

            elif (hydrogen_counts == [3, 1] or hydrogen_counts == [1, 3]) and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH3-CH-CHCl case or CH=CH-CH3 J = 7 (Douplet of quartet)
                # an approximation
                J_octet = 3.5
                coupling_pattern = [(-3.5*J_octet, 1/12), 
                                   (-2.5*J_octet, 1/12),  
                                  (-1.5*J_octet, 2/12), 
                                  (-0.5*J_octet, 2/12), 
                                  (0.5*J_octet, 2/12), 
                                  (1.5*J_octet, 2/12), 
                                  (2.5*J_octet, 1/12),
                                  (3.5*J_octet, 1/12)]


            elif (hydrogen_counts == [3, 2] or hydrogen_counts == [2, 3]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH3-CH2-CH2 case (Quartet of Triplets) most likely like a sextet
                # an approximation
                J_sixtet = 7
                coupling_pattern = [(-2.5*J_sixtet, 2/12), 
                                      (-1.5*J_sixtet, 4/12), 
                                      (-0.5*J_sixtet, 6/12), 
                                      (0.5*J_sixtet, 6/12),
                                      (1.5*J_sixtet, 4/12), 
                                      (2.5*J_sixtet, 2/12)]

            elif hydrogen_counts == [3, 3]:
                # CH3-CH-CH3 with another connection to CH with CH3 J= 6.4
                J_septet = 6.4  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                coupling_pattern = [(-3*J_septet, 1/16), 
                                   (-2*J_septet, 2/16),  
                                  (-1*J_septet, 3/16), 
                                  (0*J_septet, 4/16), 
                                  (1*J_septet, 3/16), 
                                  (2*J_septet, 2/16),
                                  (3*J_septet, 1/16)]
            elif hydrogen_counts == [0, 0, 0]:
                coupling_pattern = [(0, 1)]  

            elif hydrogen_counts == [0, 0, 1] or hydrogen_counts == [0, 1, 0] or hydrogen_counts == [1, 0, 0]:
                # (CCl3)2-CH-CHCl2
                J_doublet = 6.1 
                coupling_pattern = [(-0.5*J_doublet, 0.5), 
                                  (0.5*J_doublet, 0.5)]  
                
            elif (hydrogen_counts == [0, 1, 1] or hydrogen_counts == [1,1,0] or hydrogen_counts == [1,0,1]):
                #COCl-CH-(CHCl)2
                J_triplet = 7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/4), 
                                      (0, 2/4), 
                                      (J_triplet, 1/4)]   
                
            elif (hydrogen_counts == [0,2,2] or hydrogen_counts == [2,2,0] or hydrogen_counts == [2,0,2]):
                #COCl-CH-(CH2)
                J_quintet = 7.5
                coupling_pattern = [(-2 * J_quintet, 1/9), 
                                    (-J_quintet, 2/9), 
                                    (0, 3/9), 
                                    (J_quintet, 2/9), 
                                    (2 * J_quintet, 1/9)]
            elif (hydrogen_counts ==  [0, 2, 0] 
                   or hydrogen_counts ==  [0, 0, 2] 
                   or hydrogen_counts ==  [2, 0, 0]):
                # Approximation dd ->t
                J_triplet = 7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/4), 
                                      (0, 2/4), 
                                      (J_triplet, 1/4)]   
            
            elif (hydrogen_counts ==  [0, 2, 1] 
                   or hydrogen_counts ==  [0, 1, 2] 
                   or hydrogen_counts ==  [1, 2, 0]
                   or hydrogen_counts ==  [1, 0, 2]
                   or hydrogen_counts ==  [2, 0, 1]
                   or hydrogen_counts ==  [2, 1, 0]):
                #COCl-CH-(CH2)(CH) ddd
                # an approximation
                J_quartet = 7.0   # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]

            elif (hydrogen_counts == [1,2,2]
                   or hydrogen_counts ==  [2, 1, 2] 
                   or hydrogen_counts ==  [2, 2, 1]):
                    #CH-CH-(CH2)2  ttd
                J_septet = 6.2  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                J_12 = 3
                coupling_pattern = [(-5.5*J_12, 1/42), 
                                   (-4.5*J_12, 2/42),  
                                  (-3.5*J_12, 3/42), 
                                  (-2.5*J_12, 4/42), 
                                  (-1.5*J_12, 5/42), 
                                  (-0.5*J_12, 6/42),
                                  (0.5*J_12, 6/42), 
                                  (1.5*J_12, 5/42), 
                                  (2.5*J_12, 4/42),
                                  (3.5*J_12, 3/42), 
                                  (4.5*J_12, 2/42), 
                                  (5.5*J_12, 1/42)]   
                
            elif hydrogen_counts == [2,2,2]:
                #CH2-CH-(CH2)2
                J_septet = 6.2  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                coupling_pattern = [(-3*J_septet, 1/16), 
                                   (-2*J_septet, 2/16),  
                                  (-1*J_septet, 3/16), 
                                  (0*J_septet, 4/16), 
                                  (1*J_septet, 3/16), 
                                  (2*J_septet, 2/16),
                                  (3*J_septet, 1/16)]            
            try:
                if hydrogen_counts != None:
                    coupling_patterns.append(coupling_pattern)
                    atoms_done.append(atom_data['label'])
                    shifts.append(atom_data['shift'])
                    hydrogen_num.append(atom_data['neighbors']+1)
                    atom_indices.append(atom_data['atom_index'])
                    parent_atom_indices.append(atom_data['parent_atom_index'])  # Add this line

                else:
                    continue
            except:
                print(hydrogen_counts, n_neighbors)

        #if atom_data["label"] =="C1H1":
        #    break
    return coupling_patterns, atoms_done, shifts, hydrogen_num, atom_indices, parent_atom_indices  # Add parent_atom_indices to the return values

def get_multiplicity_label(coupling_pattern, n_neighbors):
    """
    Convert coupling pattern to standard NMR multiplicity notation
    Returns tuple of (label, number of hydrogens affected)
    """
    # Count number of unique peaks (accounting for very close peaks)
    unique_shifts = set()
    for shift, _ in coupling_pattern:
        rounded = round(shift, 1)  # Round to handle floating point comparison
        unique_shifts.add(rounded)
    
    peak_count = len(unique_shifts)
    
    # Special cases first
    if peak_count == 1:
        return 's'  # singlet
    elif peak_count == 2:
        return 'd'  # doublet
    elif peak_count == 3:
        return 't'  # triplet
    elif peak_count == 4:
        return 'q'  # quartet
    elif peak_count == 5:
        return 'p'  # pentet/quintet
    elif peak_count == 6:
        return 'h'  # hextet/sextet
    elif peak_count == 7:
        return 'sept'  # septet
    elif peak_count == 8:
        return 'o'  # octet
    elif peak_count > 8:
        return 'm'  # multiplet
    return 'm'  # default to multiplet if pattern isn't recognized

def calculate_couplings_constants_new(nmr_data):
    
    J_aromatic = 8.0

    ### Version 2
    # Calculate coupling patterns using the average shifts
    atoms_done = []
    coupling_patterns = []
    multiplicity_patterns = []  # New list for multiplicity labels
    hydrogen_num = []
    shifts = []
    hydrogen_counts = None
    atom_indices = []
    parent_atom_indices = []

    for atom_data in nmr_data:
        if ("N" in atom_data["label"] or "O" in atom_data["label"]):
            continue
        parent_atom = atom_data['atom'].GetNeighbors()[0]
        if (atom_data['aromatic'] and atom_data['label'] not in atoms_done):
            n_neighbors = atom_data['neighbors']
            adjacent_aromatic_hydrogens = get_adjacent_aromatic_hydrogens(parent_atom)
            arom_n_neighbors = len(adjacent_aromatic_hydrogens)
            if arom_n_neighbors == 0:
                coupling_pattern = [(J_aromatic, 1)]
            else:
                coupling_pattern = generate_nmr_coupling_pattern(arom_n_neighbors, J_aromatic)
            coupling_patterns.append(coupling_pattern)
            multiplicity_patterns.append([get_multiplicity_label(coupling_pattern, n_neighbors), 
                                       atom_data['neighbors']+1])
            shifts.append(atom_data['shift'])
            atoms_done.append(atom_data['label'])
            hydrogen_num.append(atom_data['neighbors']+1)
            atom_indices.append(atom_data['atom_index'])
            parent_atom_indices.append(atom_data['parent_atom_index'])
            
        elif atom_data['label'] not in atoms_done:

            bond_types = [bond.GetBondType() for bond in parent_atom.GetBonds() if bond.GetOtherAtom(parent_atom).GetSymbol() == 'C']        

            n_neighbors = atom_data['neighbors']

            carbon_neighbors = [neighbor for neighbor in parent_atom.GetNeighbors() if neighbor.GetSymbol() == 'C']

            hydrogen_counts = [sum(1 for neighbor in carbon_neighbor.GetNeighbors() if neighbor.GetSymbol() == 'H') for carbon_neighbor in carbon_neighbors]
            #print(hydrogen_counts, atom_data["label"])
            # Rule-based coupling pattern generation
            if hydrogen_counts == [] and n_neighbors == 2:
                #N-CH3
                coupling_pattern = [(0, 3)]   

            if hydrogen_counts == [] and n_neighbors == 1:
                #CCl2=CH2
                coupling_pattern = [(0, 2)]  
                
            if hydrogen_counts == [] and n_neighbors == 0:
                #(CCl2)3-CH
                coupling_pattern = [(0, 0)]  

            if hydrogen_counts == [0] and n_neighbors == 2:
                coupling_pattern = [(0, 3)]

            if hydrogen_counts == [0] and n_neighbors == 1:
                coupling_pattern = [(0, 2)]   

            if hydrogen_counts == [0] and n_neighbors == 0:
                coupling_pattern = [(0, 1)]  


            if hydrogen_counts == [0, 0] and n_neighbors == 0:
                coupling_pattern = [(0, 1)]  
            
            if hydrogen_counts == [0, 0] and n_neighbors == 1:
                coupling_pattern = [(0, 2)]   

            if hydrogen_counts == [1] and Chem.rdchem.BondType.DOUBLE in bond_types and n_neighbors == 1:
                # CH=CH2 case J = 16 10
                ### Approximation
                J_doublet_1 = 16  
                J_doublet_2 = 10 
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/2), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/2),
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/2), 
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/2)]

            elif hydrogen_counts == [1] and Chem.rdchem.BondType.SINGLE in bond_types and n_neighbors == 1:
                # CH-CH2-Cl case J=5.9
                J_doublet = 5.9  # Coupling constant for the single bond between the CH hydrogens
                coupling_pattern = [(-0.5*J_doublet, 1), 
                                      (0.5*J_doublet, 1)]

            elif hydrogen_counts == [1] and Chem.rdchem.BondType.SINGLE in bond_types and n_neighbors == 2:
                # CH-CH3 case J = 6.1
                J_doublet = 6.1  # Coupling constant for the single bond between the CH hydrogens
                coupling_pattern = [(-0.5*J_doublet, 1.5), 
                                      (0.5*J_doublet, 1.5)]
                
            if hydrogen_counts == [1] and n_neighbors == 0:
                #(CCl2)2-CH-CHCl2                
                J_doublet = 6.1
                coupling_pattern = [(-0.5*J_doublet, 0.5),
                                    (0.5*J_doublet, 0.5)]  
                
            #elif hydrogen_counts == [2] and Chem.rdchem.BondType.DOUBLE in bond_types:
            #    # CH2=CH2 case
            #    J_triplet = J_double_bond  # Coupling constant for the double bond between the CH2=CH2 hydrogens
            #    coupling_pattern = [(-J_triplet, 1/2), 
            #                          (0, 2/2), 
            #                          (J_triplet, 1/2)]

            elif hydrogen_counts == [2] and Chem.rdchem.BondType.SINGLE in bond_types:
                # CH2-CH2 case J = 6.3
                J_triplet = 6.3  # Coupling constant for the single bond between the CH2-CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]

            elif hydrogen_counts == [3] and Chem.rdchem.BondType.SINGLE in bond_types:
                # CH3-CH2 case J = 7
                J_quartet =   7.0
                coupling_pattern = [(-1.5*J_quartet, 2/6), 
                                      (-0.5*J_quartet, 4/6), 
                                      (0.5*J_quartet, 4/6), 
                                      (1.5*J_quartet, 2/6)]

            elif (hydrogen_counts == [1,0] or hydrogen_counts == [0,1]):
                # CH-CH2-CO case J = 7
                J_doublet = 6.9  
                coupling_pattern = [(-0.5*J_doublet, 1), 
                                      (0.5*J_doublet, 1)]            


            elif (hydrogen_counts == [2,0] or hydrogen_counts == [0,2]) and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH2=CH-CO case J = 7
                J_doublet_1 = 18  # Coupling constant for the double bond between the CH=CH2 hydrogens
                J_doublet_2 = 10  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/4),
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/4)]  

            elif (hydrogen_counts == [2,0] or hydrogen_counts == [0,2]) and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH2-CH2-CO case J = 7
                J_triplet = 6.7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]   

            elif hydrogen_counts == [3,0]  or hydrogen_counts == [0,3]:
                # CH3-CHCl-CO case J = 7
                J_quartet = 7  
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]
                
            elif hydrogen_counts == [1, 1] and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH-CH=CH non-aromatic case  13/6.06
                J_doublet_1 = 6.06  # Coupling constant for the single bond between CH hydrogens
                J_doublet_2 = 13 # Coupling constant for the double bond between the CH=CH hydrogens
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/4),
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/4)]

            elif hydrogen_counts == [1, 1] and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH-CH2-CH non-aromatic case  13/6.06
                J_triplet = 6.0  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]   

            elif (hydrogen_counts == [1, 2] or hydrogen_counts == [2, 1]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 1 and bond_types.count(Chem.rdchem.BondType.DOUBLE) == 1:  
                # CH=CH-CH2 case  J = 7.4 
                J_quartet = 7.4  # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]

            elif (hydrogen_counts == [1, 2] or hydrogen_counts == [2, 1]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH-CH2-CH2 case  J = 7.4 
                # an approximation
                J_quartet = 7.4   # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 2/6), 
                                      (-0.5*J_quartet, 4/6), 
                                      (0.5*J_quartet, 4/6), 
                                      (1.5*J_quartet, 2/6)]

            elif hydrogen_counts == [2, 2] and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH2-CH2-CH2 case (quintet)  J=6.57
                J_quintet = 6.57
                coupling_pattern = [(-2 * J_quintet, 2/9), 
                                    (-J_quintet, 4/9), 
                                    (0, 6/9), 
                                    (J_quintet, 4/9), 
                                    (2 * J_quintet, 2/9)]

            elif (hydrogen_counts == [3, 1] or hydrogen_counts == [1, 3]) and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH3-CH=CH case or CH=CH-CH3 J = 7 (Douplet of quartet)
                # an approximation
                J_quintet = 7
                coupling_pattern = [(-2 * J_quintet, 1/9), 
                                    (-J_quintet, 2/9), 
                                    (0, 3/9), 
                                    (J_quintet, 2/9), 
                                    (2 * J_quintet, 1/9)]

            elif (hydrogen_counts == [3, 1] or hydrogen_counts == [1, 3]) and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH3-CH-CHCl case or CH=CH-CH3 J = 7 (Douplet of quartet)
                # an approximation
                J_octet = 3.5
                coupling_pattern = [(-3.5*J_octet, 1/12), 
                                   (-2.5*J_octet, 1/12),  
                                  (-1.5*J_octet, 2/12), 
                                  (-0.5*J_octet, 2/12), 
                                  (0.5*J_octet, 2/12), 
                                  (1.5*J_octet, 2/12), 
                                  (2.5*J_octet, 1/12),
                                  (3.5*J_octet, 1/12)]


            elif (hydrogen_counts == [3, 2] or hydrogen_counts == [2, 3]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH3-CH2-CH2 case (Quartet of Triplets) most likely like a sextet
                # an approximation
                J_sixtet = 7
                coupling_pattern = [(-2.5*J_sixtet, 2/12), 
                                      (-1.5*J_sixtet, 4/12), 
                                      (-0.5*J_sixtet, 6/12), 
                                      (0.5*J_sixtet, 6/12),
                                      (1.5*J_sixtet, 4/12), 
                                      (2.5*J_sixtet, 2/12)]

            elif hydrogen_counts == [3, 3]:
                # CH3-CH-CH3 with another connection to CH with CH3 J= 6.4
                J_septet = 6.4  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                coupling_pattern = [(-3*J_septet, 1/16), 
                                   (-2*J_septet, 2/16),  
                                  (-1*J_septet, 3/16), 
                                  (0*J_septet, 4/16), 
                                  (1*J_septet, 3/16), 
                                  (2*J_septet, 2/16),
                                  (3*J_septet, 1/16)]
            elif hydrogen_counts == [0, 0, 0]:
                coupling_pattern = [(0, 1)]  

            elif hydrogen_counts == [0, 0, 1] or hydrogen_counts == [0, 1, 0] or hydrogen_counts == [1, 0, 0]:
                # (CCl3)2-CH-CHCl2
                J_doublet = 6.1 
                coupling_pattern = [(-0.5*J_doublet, 0.5), 
                                  (0.5*J_doublet, 0.5)]  
                
            elif (hydrogen_counts == [0, 1, 1] or hydrogen_counts == [1,1,0] or hydrogen_counts == [1,0,1]):
                #COCl-CH-(CHCl)2
                J_triplet = 7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/4), 
                                      (0, 2/4), 
                                      (J_triplet, 1/4)]   
                
            elif (hydrogen_counts == [0,2,2] or hydrogen_counts == [2,2,0] or hydrogen_counts == [2,0,2]):
                #COCl-CH-(CH2)
                J_quintet = 7.5
                coupling_pattern = [(-2 * J_quintet, 1/9), 
                                    (-J_quintet, 2/9), 
                                    (0, 3/9), 
                                    (J_quintet, 2/9), 
                                    (2 * J_quintet, 1/9)]
            elif (hydrogen_counts ==  [0, 2, 0] 
                   or hydrogen_counts ==  [0, 0, 2] 
                   or hydrogen_counts ==  [2, 0, 0]):
                # Approximation dd ->t
                J_triplet = 7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/4), 
                                      (0, 2/4), 
                                      (J_triplet, 1/4)]   
            
            elif (hydrogen_counts ==  [0, 2, 1] 
                   or hydrogen_counts ==  [0, 1, 2] 
                   or hydrogen_counts ==  [1, 2, 0]
                   or hydrogen_counts ==  [1, 0, 2]
                   or hydrogen_counts ==  [2, 0, 1]
                   or hydrogen_counts ==  [2, 1, 0]):
                #COCl-CH-(CH2)(CH) ddd
                # an approximation
                J_quartet = 7.0   # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]

            elif (hydrogen_counts == [1,2,2]
                   or hydrogen_counts ==  [2, 1, 2] 
                   or hydrogen_counts ==  [2, 2, 1]):
                    #CH-CH-(CH2)2  ttd
                J_septet = 6.2  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                J_12 = 3
                coupling_pattern = [(-5.5*J_12, 1/42), 
                                   (-4.5*J_12, 2/42),  
                                  (-3.5*J_12, 3/42), 
                                  (-2.5*J_12, 4/42), 
                                  (-1.5*J_12, 5/42), 
                                  (-0.5*J_12, 6/42),
                                  (0.5*J_12, 6/42), 
                                  (1.5*J_12, 5/42), 
                                  (2.5*J_12, 4/42),
                                  (3.5*J_12, 3/42), 
                                  (4.5*J_12, 2/42), 
                                  (5.5*J_12, 1/42)]   
                
            elif hydrogen_counts == [2,2,2]:
                #CH2-CH-(CH2)2
                J_septet = 6.2  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                coupling_pattern = [(-3*J_septet, 1/16), 
                                   (-2*J_septet, 2/16),  
                                  (-1*J_septet, 3/16), 
                                  (0*J_septet, 4/16), 
                                  (1*J_septet, 3/16), 
                                  (2*J_septet, 2/16),
                                  (3*J_septet, 1/16)] 
            try:
                if hydrogen_counts != None:
                    coupling_patterns.append(coupling_pattern)
                    multiplicity_patterns.append([get_multiplicity_label(coupling_pattern, n_neighbors),
                                               atom_data['neighbors']+1])
                    atoms_done.append(atom_data['label'])
                    shifts.append(atom_data['shift'])
                    hydrogen_num.append(atom_data['neighbors']+1)
                    atom_indices.append(atom_data['atom_index'])
                    parent_atom_indices.append(atom_data['parent_atom_index'])
                else:
                    continue
            except:
                print(hydrogen_counts, n_neighbors)

    return coupling_patterns, multiplicity_patterns, atoms_done, shifts, hydrogen_num, atom_indices, parent_atom_indices


def create_plot_NMR(shifts, coupling_patterns, gamma, spectrometer_frequency):
    x, y = simulate_splitting(np.array(shifts), coupling_patterns, gamma, spectrometer_frequency)
    plt.plot(x, y)
    plt.xlabel('Chemical shift (ppm)')
    plt.ylabel('Intensity')
    for shift, label in zip(shifts, atoms_done):
        plt.text(shift, np.max(lorentzian(x, shift, gamma)), label, ha='center', va='bottom', fontsize=8, rotation=45)

    plt.gca().invert_xaxis()
    plt.show()
    
def create_plot_NMR_interactiv(shift_intensity_label_data):
    
    # Create an interactive plot using Plotly
    fig = go.Figure()

    for shift, intensity, label in shift_intensity_label_data:
        fig.add_trace(
            go.Scatter(
                x=[shift, shift],  # Use two points to create a vertical line
                y=[0, intensity],
                mode="lines",
                line=dict(color="black", width=1.5),
                hoverinfo="none",  # Disable hover info
            )
        )

    # Find the maximum intensity for each shift
    max_intensities_dict = {}
    for item in shift_intensity_label_data:
        _, intensity, label = item
        if label not in max_intensities_dict:
            max_intensities_dict[label] = intensity
        else:
            max_intensities_dict[label] = max(max_intensities_dict[label], intensity)
    max_intensities = list(max_intensities_dict.values())


    # Add a separate trace for the labels
    fig.add_trace(
        go.Scatter(
            x=shifts,
            y=max_intensities,
            mode="text",
            text=atoms_done,
            textposition="top center",
            hoverinfo="none",  # Disable hover info
        )
    )

    fig.update_layout(
        xaxis=dict(title="Chemical shift (ppm)", range=[11, 0]),  # Set x-axis range
        yaxis=dict(title="Intensity"),
        showlegend=False,  # Remove the legend from the plot
    )

    multiplicity_labels = []
    for coupling_pattern in coupling_patterns:
        if len(coupling_pattern) == 1:
            multiplicity_labels.append("Singlet")
        else:
            n_split = len(coupling_pattern)
            if n_split == 2:
                multiplicity_labels.append("Doublet")
            elif n_split == 3:
                multiplicity_labels.append("Triplet")
            elif n_split == 4:
                multiplicity_labels.append("Quartet")
            elif n_split == 5:
                multiplicity_labels.append("Quintet")
            elif n_split>5:
                multiplicity_labels.append(f"{n_split} peaks")

    fig.add_trace(
        go.Scatter(
            x=shifts,
            y=[intensity - 0.1 * intensity for intensity in max_intensities],
            mode="text",
            text=multiplicity_labels,
            textposition="bottom center",
            textfont=dict(color="red"),  # Use a different color for multiplicity labels
            hoverinfo="none",  # Disable hover info
        )
    )
    fig.show()
    
def create_labeled_structure(mol,assigned_shifts):


    # Generate a 2D depiction of the molecule to better fit the drawing canvas
    rdDepictor.Compute2DCoords(mol)

    # Create a MolDraw2DSVG object to draw the molecule as an SVG
    # You can adjust the width and height values to better fit the molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 200)
    opts = drawer.drawOptions()

    # Set atom labels based on the assigned_shifts dictionary
    for (atom_type, atom_idx) in assigned_shifts.keys():
        atom = mol.GetAtomWithIdx(atom_idx)
        opts.atomLabels[atom_idx] = f"{atom_type}{atom_idx}"

    # Draw the molecule
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Display the SVG in the notebook
    svg = SVG(drawer.GetDrawingText())
    return svg                     
                             
def create_shift_intensity_label_data(shifts, coupling_patterns, atoms_done, atom_indices, parent_atom_indices, spectrometer_frequency):
    shift_intensity_label_data = []
    shift_intensity_data = []
    for shift, coupling_pattern, label, atom_index, parent_atom_index in zip(shifts, coupling_patterns, atoms_done, atom_indices, parent_atom_indices):
        if shift != 0.0:
            for J, intensity in coupling_pattern:
                if len(coupling_pattern) > 1:
                    shift1 = shift + (J / spectrometer_frequency)
                else:
                    shift1 = shift
                #import IPython; IPython.embed();
                shift_intensity_label_data.append((shift1, intensity, label, atom_index, parent_atom_index))
                shift_intensity_data.append((shift1, intensity, atom_index, parent_atom_index))
    return shift_intensity_label_data, shift_intensity_data


def run_1H_generation(sdf_file_path):

    ### Settings
    spectrometer_frequency = 400  # Example spectrometer frequency in MHz
    gamma = 0.01
    plot_NMR = False
    plot_NMR_interactiv = False
    show_labeled_structure = False

    try:
        mol = load_mol_and_assign_shifts(sdf_file_path)
        file_name = os.path.basename(sdf_file_path)
        sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

        nmr_data, assigned_shifts, mol = analyze_molecule(mol)
        nmr_data = add_shifts_to_data(nmr_data, assigned_shifts)
        #import IPython; IPython.embed();
        coupling_patterns, multiplicity_patterns, atoms_done, shifts, hydrogen_num, atom_indices, parent_atom_indices = calculate_couplings_constants_new(nmr_data)
        shift_intensity_label_data, shift_intensity_data = create_shift_intensity_label_data(shifts, coupling_patterns, atoms_done, atom_indices, parent_atom_indices, spectrometer_frequency)
        #print("run_1H_generation")
        #import IPython; IPython.embed();

        if len(shift_intensity_data) != 0:
            mol = Chem.RemoveHs(mol)
            smi = MolToSmiles(mol)
            
            # Create DataFrame
            df_1h = pd.DataFrame(shift_intensity_data, columns=['shifts', 'intensity', 'H_atom_index', 'parent_atom_index'])
            # Create expanded multiplicity patterns that match the intensity data
            
           # Pad multiplicity_patterns with None to match DataFrame length
            padded_multiplicity = multiplicity_patterns + [None] * (len(df_1h) - len(multiplicity_patterns))
            df_1h['multiplicity_patterns_and_num_hydrogens'] = padded_multiplicity
            
            # Pad shifts with None while keeping original values
            padded_shifts = shifts + [None] * (len(df_1h) - len(shifts))

            parent_atom_indices_padded = parent_atom_indices + [None] * (len(df_1h) - len(parent_atom_indices))

            df_1h['parent_atom_indices'] = parent_atom_indices_padded
            df_1h['shifts_orig'] = padded_shifts
            
            df_1h['SMILES'] = smi
            df_1h['sample-id'] = sample_id
            return df_1h

        else:
            print(f"No data for sample {sample_id}")
            return None

    except Exception as e:
        print(f"Error processing file {sdf_file_path}: {str(e)}")
        return None



############################################################################################
###################################### EXTRA ###############################################
############################################################################################


"""
def save_smiles_as_csv(smiles, output_dir):
    # Generate a random sample ID
    mol = Chem.MolFromSmiles(smiles)

    isomers = tuple(EnumerateStereoisomers(mol))
    smiles = Chem.MolToSmiles(isomers[0],  isomericSmiles=False, canonical=True)  
    
    sample_id = f"GEN{random.randint(0, 999999):06d}"
    
    # Create the CSV filename
    csv_filename = f"{sample_id}.csv"
    
    # Construct the full path
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Write the data to the CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SMILES', 'sample-id'])  # Header
        writer.writerow([smiles, sample_id])  # Data
    
    return csv_path"""

def save_smiles_as_csv(smiles_input, output_dir):
    # Check if the input is a single SMILES or a list
    if isinstance(smiles_input, str):
        smiles_list = [smiles_input]  # Convert single SMILES string to a list
    elif isinstance(smiles_input, list):
        smiles_list = smiles_input
    else:
        raise ValueError("Input should be either a SMILES string or a list of SMILES strings.")
    
    csv_filename = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SMILES', 'sample-id'])  # Header
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                isomers = tuple(EnumerateStereoisomers(mol))
                canonical_smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=False, canonical=True)
                sample_id = f"GEN{random.randint(0, 999999):06d}"
                writer.writerow([canonical_smiles, sample_id])
            except:
                print(f"Error processing SMILES: {smiles}")
    
    return csv_path

def combine_nmr_data(sdf_path, df_13c, df_1h, df_hsqc, df_cosy):
    
    # Define column mappings
    c13_cols = {'atom': 'atom_index', 'shift': 'shifts', 'sample-id': 'sample-id'}
    h1_cols = {'atom': 'H_atom_index', 'parent': 'parent_atom_index', 'shift': 'shifts', 'intensity': 'intensity', 'sample-id': 'sample-id'}
    hsqc_cols = {'c_atom': 'atom_index', 'c_shift': 'F1 (ppm)', 'h_shift': 'F2 (ppm)', 'sample-id': 'sample-id'}
    cosy_cols = {'atom1': 'atom_index_1', 'atom2': 'atom_index_2', 'shift1': 'F1 (ppm)', 'shift2': 'F2 (ppm)', 'sample-id': 'sample-id'}
    
    # Read SMILES from SDF file
    mol = Chem.SDMolSupplier(sdf_path)[0]
    
    isomers = tuple(EnumerateStereoisomers(mol))
    smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=False, canonical=True)  
    
    nmr_data = {
        smiles: {
            "13C": {},
            "1H": {},
            "HSQC": {},
            "COSY": {}
        }
    }
    
    # Initialize example_molecules list
    example_molecules = []
    
    # Process 13C data
    for _, row in df_13c.iterrows():
        nmr_data[smiles]["13C"][row[c13_cols['atom']]] = row[c13_cols['shift']]
        if (smiles, row[c13_cols['sample-id']]) not in example_molecules:
            example_molecules.append((smiles, row[c13_cols['sample-id']]))
    
    # Process 1H data
    for _, row in df_1h.iterrows():
        parent_idx = row[h1_cols['parent']]
        if parent_idx not in nmr_data[smiles]["1H"]:
            nmr_data[smiles]["1H"][parent_idx] = []
        nmr_data[smiles]["1H"][parent_idx].append((row[h1_cols['shift']], row[h1_cols['intensity']]))
        if (smiles, row[h1_cols['sample-id']]) not in example_molecules:
            example_molecules.append((smiles, row[h1_cols['sample-id']]))
    
    # Process HSQC data
    for _, row in df_hsqc.iterrows():
        nmr_data[smiles]["HSQC"][row[hsqc_cols['c_atom']]] = (row[hsqc_cols['c_shift']], row[hsqc_cols['h_shift']])
        if (smiles, row[hsqc_cols['sample-id']]) not in example_molecules:
            example_molecules.append((smiles, row[hsqc_cols['sample-id']]))
    
    # Process COSY data
    for _, row in df_cosy.iterrows():
        atom1 = row[cosy_cols['atom1']]
        if atom1 not in nmr_data[smiles]["COSY"]:
            nmr_data[smiles]["COSY"][atom1] = []
        nmr_data[smiles]["COSY"][atom1].append((row[cosy_cols['shift1']], row[cosy_cols['shift2']]))
        if (smiles, row[cosy_cols['sample-id']]) not in example_molecules:
            example_molecules.append((smiles, row[cosy_cols['sample-id']]))
    
    return nmr_data, example_molecules