##############################################################################
################################ SGNN ########################################
##############################################################################

# Standard library imports
import collections
import glob
import os
import random
import shutil
import time
from tqdm import tqdm
import collections
from IPython.display import display, SVG
from torch.utils.data import Dataset, DataLoader

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotly.graph_objs as go
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, rdmolfiles, MolToSmiles, SDMolSupplier, Descriptors, PandasTools, AddHs, rdDepictor
from rdkit.Chem import EnumerateStereoisomers
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount, EnumerateStereoisomers

# Local application/library specific imports
from utils_MMT.sgnn_code_pl_v15_4 import *
import utils_MMT.dataloaders_pl_v15_4 as dl
# import utils_MMT.clip_functions_v15_4 as cl
import utils_MMT.nmr_calculation_from_dft_v15_4 as ncfd



def contains_hydrogen_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'H':
                    return True
        return False
    except:
        return False

# Calculate molecular weight
def calculate_mw(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.MolWt(mol) if mol is not None else None
    except:
        return None

tqdm.pandas()

# Function to check if a SMILES has disconnected structures
def has_disconnected_smiles(smiles_string):
    return '.' in smiles_string



def run_sgnn(config):
    graph_representation = "sparsified"
    target = "13C"
    train_y_mean_C, train_y_std_C = load_std_mean(target,graph_representation)
    target = "1H"
    train_y_mean_H, train_y_std_H = load_std_mean(target,graph_representation)
    sgnn_means_stds = (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)


    batch_size = 128
    path_csv = config.SGNN_csv_gen_smi
    ML_save_folder = config.SGNN_gen_folder_path
    data_df = pd.read_csv(path_csv)
    #data_df = data_df.iloc[38449:]
    ### Remove every molecule that does not have any hydrogens
    data_df = data_df[data_df['SMILES'].apply(contains_hydrogen_rdkit)]
    # Filter out disconnected structures
    data_df = data_df[~data_df['SMILES'].apply(has_disconnected_smiles)]  # Note the use of ~ for negation

    data_df['Molecular_Weight'] = data_df['SMILES'].apply(calculate_mw)
    data_df_final = data_df[data_df['Molecular_Weight'] <= config.SGNN_size_filter]

    #data_df['SMILES'] = data_df['smiles']
    #data_df['sample-id'] = data_df['zinc_id']
    #data_df = data_df.iloc[:]#


    if not os.path.exists(ML_save_folder):
        os.mkdir(ML_save_folder)

    batch_data_1, failed_ids_1 = main_execute(data_df_final, sgnn_means_stds, ML_save_folder, batch_size)

    print(len(failed_ids_1))

    # second round just to let out the failed molecules
    batch_size = 1
    data_df_final = data_df_final[data_df_final['sample-id'].isin(failed_ids_1)]
    batch_data_2, failed_ids_2 = main_execute(data_df_final, sgnn_means_stds, ML_save_folder, batch_size)
    batch_data_1, batch_data_2
    combined_df = pd.concat([batch_data_1, batch_data_2], axis=0, ignore_index=True)
    return combined_df



##############################################################################
################################ 1H ##########################################
##############################################################################

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

# Updated analyze_molecule function to combine symmetric hydrogen shifts
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
    ### Calculate the average shift to the nmr_data dictionary 
    ### where multiple H are attached to one Carbon
    grouped_shifts = {}
    for atom_data in nmr_data:
        group_key = atom_data['group_key']
        if group_key not in grouped_shifts:
            grouped_shifts[group_key] = []
        if group_key in assigned_shifts:  
            # Check if the key is in assigned_shifts before appending
            grouped_shifts[group_key].append(assigned_shifts[group_key])
    
    # Take the average of shifts in each group
    avg_shifts = {group_key: np.mean([shift for shift in shifts if np.isfinite(shift)]) for group_key, shifts in grouped_shifts.items()}

    # Replace the original shifts with the average shifts
    for atom_data in nmr_data:
        group_key = atom_data['group_key']
        if group_key in avg_shifts:  
            # Check if the key is in avg_shifts before assigning
            atom_data['shift'] = avg_shifts[group_key]
            ### This creates duplicates entries for the hydrogens that are on the same Carbon
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
    
    for atom_data in nmr_data:
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
                else:
                    continue
            except:
                print(hydrogen_counts, n_neighbors)

        #if atom_data["label"] =="C1H1":
        #    break
    return coupling_patterns, atoms_done, shifts, hydrogen_num


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
                             
def create_shift_intensity_label_data(shifts, coupling_patterns, atoms_done, spectrometer_frequency):
    shift_intensity_label_data = []
    shift_intensity_data = []
    for shift, coupling_pattern, label in zip(shifts, coupling_patterns, atoms_done):
        if shift!=0.0:   ### Because of ACD labs sometimes there are 0.0 that we used for padding to the right number. these should be ignored
            for J, intensity in coupling_pattern:
                if len(coupling_pattern) > 1:
                    shift1 = shift + (J / spectrometer_frequency)
                else:
                    shift1 = shift
                shift_intensity_label_data.append((shift1, intensity, label))
                shift_intensity_data.append((shift1, intensity))
    return shift_intensity_label_data, shift_intensity_data


def run_1H_generation(config):
    folder_path = config.SGNN_gen_folder_path
    nmr_files = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    nmr_files = sorted(nmr_files, reverse=False)


    ### Settings
    spectrometer_frequency = 400  # Example spectrometer frequency in MHz
    gamma = 0.01
    plot_NMR = False
    plot_NMR_interactiv = False
    show_labeled_structure = False

    ### Stores
    smiles_list = []
    data_list = []
    sample_id_list = []

    for file_path in nmr_files[:]:
        try:
            mol = load_mol_and_assign_shifts(file_path)
            #sample_id = file_path[-19:-4]

            # Extract the filename from the path
            file_name = os.path.basename(file_path)

            # Remove the file extension to get the desired part
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            #sample_id = file_path[-17:-4] ### For zinc dataset
            nmr_data, assigned_shifts, mol = analyze_molecule(mol)
            nmr_data = add_shifts_to_data(nmr_data, assigned_shifts)
            coupling_patterns, atoms_done, shifts, hydrogen_num = calculate_couplings_constants(nmr_data)
            if plot_NMR:
                create_plot_NMR(shifts, coupling_patterns, gamma, spectrometer_frequency)
            shift_intensity_label_data, shift_intensity_data = create_shift_intensity_label_data(shifts, coupling_patterns, atoms_done, spectrometer_frequency)
            if plot_NMR_interactiv:
                create_plot_NMR_interactiv(shift_intensity_label_data)
            if show_labeled_structure:
                svg_output = create_labeled_structure(mol,assigned_shifts)
                display(svg_output)
            if len(shift_intensity_data)!=0:
                mol = Chem.RemoveHs(mol)
                smi = MolToSmiles(mol)            
                smiles_list.append(smi)
                ### Use a set because symmetric H on C will come for every atom the same
                data_shifts_ints = sorted(list(set(shift_intensity_data)), reverse=False)
                data_list.append(data_shifts_ints)
                sample_id_list.append(sample_id)
            else:
                print(sample_id)
        except:
            #import IPython; IPython.embed();
            print(file_path)


    # Create a DataFrame with the lists as columns
    data = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    data.reset_index(drop=True, inplace=True)
    csv_1H_path = os.path.join(config.SGNN_csv_save_folder, f"data_1H_{config.ran_num}.csv")
    data.to_csv(csv_1H_path, index=False)
    return data, csv_1H_path


##############################################################################
################################ C13 #########################################
##############################################################################


def find_symmetric_positions(stereo_smi):
    """https://github.com/rdkit/rdkit/issues/1411"""

    mol = Chem.MolFromSmiles(stereo_smi)
    z=list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    matches = mol.GetSubstructMatches(mol, uniquify=False)
    
    if len(z) != len(set(z)) and len(matches) > 1:
    # if len(matches) > 1:

        # # Get a list with all the duplicate numbers
        symmetric = [item for item, count in collections.Counter(z).items() if count > 1]
        
        # Get a list of lists with the positions of the duplicates in match list
        all_duplicates = []
        for j in symmetric:
            indices = [i for i, v in enumerate(z) if v == j]
            all_duplicates.append(indices)
        
        # Get a list of lists with the positions of the duplicates
        example_match = matches[0]
        sym_dupl_lists = []
        for sub_list in all_duplicates:
            indices_list = []
            for i in sub_list:
                position = example_match[i]
                indices_list.append(position)
            sym_dupl_lists.append(indices_list)
    
    else:
        sym_dupl_lists = []
    return sym_dupl_lists


# Function to average the symmetric peaks and update the list
def consolidate_peaks(averaged_shifts, symmetric_positions):
    # Make a copy of the original list to avoid modifying it in place
    consolidated_shifts = averaged_shifts.copy()

    for positions in symmetric_positions:
        # Calculate the average for the symmetric positions
        avg_value = sum(averaged_shifts[i] for i in positions) / len(positions)
        
        # Update the peaks at these positions with the average value
        for i in positions:
            consolidated_shifts[i] = avg_value

    return consolidated_shifts


def run_13C_generation(config):
    
    folder_path = config.SGNN_gen_folder_path
    nmr_files = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    nmr_files = sorted(nmr_files, reverse=False)

    ### Stores
    smiles_list = []
    data_list = []
    sample_id_list = []


    for file_path in nmr_files[:]:
        try:
            #mol = load_mol_and_assign_shifts(file_path)
            mol = SDMolSupplier(file_path)[0]
            isomers = tuple(EnumerateStereoisomers(mol))
            stereo_smi = Chem.MolToSmiles(isomers[0],isomericSmiles=True)

            averaged_nmr_shifts = mol.GetProp('averaged_NMR_shifts')
            sample_shifts = list(map(float, averaged_nmr_shifts.split()))


            # Extract the sample_id from the path
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Remove symmetric C
            sym_dupl_lists = find_symmetric_positions(stereo_smi)
            sym_corr_nmr_shifts = consolidate_peaks(sample_shifts, sym_dupl_lists)

            # Initialize counter for non-hydrogen atoms
            non_hydrogen_count = 0
            # Loop through each atom and count non-hydrogen atoms
            for atom in mol.GetAtoms():
                if atom.GetSymbol() != 'H':
                    non_hydrogen_count += 1

            heavy_atoms_shifts = sym_corr_nmr_shifts[:non_hydrogen_count]
            # Remove zeros
            C_atoms_shifts = [x for x in heavy_atoms_shifts if x != 0]

            # Remove symmetric peaks
            C_atoms_shifts = sorted(list(set(C_atoms_shifts)), reverse=False)

            mol = Chem.RemoveHs(mol)
            smi = MolToSmiles(mol)     

            smiles_list.append(smi)
            data_list.append(C_atoms_shifts)
            sample_id_list.append(sample_id)
        except:
            print(file_path)


    # Create a DataFrame with the lists as columns
    data = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    csv_13C_path = os.path.join(config.SGNN_csv_save_folder, f"data_13C_{config.ran_num}.csv")
    data.to_csv(csv_13C_path, index=False)
    return data, csv_13C_path



##############################################################################
################################ COSY ########################################
##############################################################################

def find_chiral_centers(molecule):
    chiral_centers = []
    
    # Loop through all atoms in the molecule
    for atom in molecule.GetAtoms():
        
        # Get the atomic number and the index (atom number) of the atom
        atomic_num = atom.GetAtomicNum()
        atom_idx = atom.GetIdx()
        
        # Check for chiral tag
        chiral_tag = atom.GetChiralTag()
        
        # Carbon atoms with atomic number 6 are the most common chiral centers
        if atomic_num == 6 and chiral_tag != Chem.ChiralType.CHI_UNSPECIFIED:
            chiral_centers.append(atom_idx)
            
    return chiral_centers

from rdkit import Chem
def find_carbons_with_relevant_neighbors(molecule):
    carbon_dict = {}
    
    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_idx = atom.GetIdx()
        
        # Check if the atom is a carbon atom
        if atomic_num == 6:
            neighbor_carbons_with_hydrogens = []
            
            # Loop through the neighbors of the current atom
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                neighbor_atomic_num = neighbor.GetAtomicNum()
                
                # Check if the neighbor is also a carbon atom and both have hydrogens
                if neighbor_atomic_num == 6 and neighbor.GetTotalNumHs() > 0 and atom.GetTotalNumHs() > 0:
                    neighbor_carbons_with_hydrogens.append(neighbor_idx)
            
            # Include carbons with hydrogens, even if they don't couple with any other
            if atom.GetTotalNumHs() > 0:
                carbon_dict[atom_idx] = neighbor_carbons_with_hydrogens
                
    return carbon_dict



def find_heavy_atoms_with_hydrogens(molecule):
    heavy_atom_dict = {}
    
    # Loop through all atoms in the molecule
    for atom in molecule.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        
        # Check if the atom is a heavy atom (i.e., not hydrogen)
        if atomic_num != 1:
            num_hydrogens = atom.GetTotalNumHs()
            
            if num_hydrogens > 0:
                heavy_atom_dict[atom_idx] = num_hydrogens
                
    return heavy_atom_dict


def extract_symmetric_hydrogen_shifts(shifts, heavy_atom_dict):
    num_heavy_atoms = len(heavy_atom_dict)
    
    # Splitting the shifts into heavy atom shifts and hydrogen shifts
    heavy_atom_shifts = shifts[:num_heavy_atoms]
    hydrogen_shifts = shifts[num_heavy_atoms:]
    
    # Initialize the dictionary to store the carbon number and the shift of the attached hydrogens
    carbon_hydrogen_shifts_dict = {}
    
    # Iterate over the heavy_atom_dict in reverse order to pick the hydrogen shifts
    for carbon, num_hydrogens in sorted(heavy_atom_dict.items(), key=lambda x: x[0], reverse=True):
        # Pick the last 'num_hydrogens' from the hydrogen_shifts list
        attached_hydrogen_shifts = hydrogen_shifts[-num_hydrogens:]
        
        # Remove the picked hydrogen shifts from the list
        hydrogen_shifts = hydrogen_shifts[:-num_hydrogens]
        
        # Store in the resulting dictionary
        carbon_hydrogen_shifts_dict[carbon] = attached_hydrogen_shifts
    
    return carbon_hydrogen_shifts_dict


def find_symmetric_positions(stereo_smi):
    """https://github.com/rdkit/rdkit/issues/1411"""

    mol = Chem.MolFromSmiles(stereo_smi)
    z=list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    matches = mol.GetSubstructMatches(mol, uniquify=False)
    
    if len(z) != len(set(z)) and len(matches) > 1:
    # if len(matches) > 1:

        # # Get a list with all the duplicate numbers
        symmetric = [item for item, count in collections.Counter(z).items() if count > 1]
        
        # Get a list of lists with the positions of the duplicates in match list
        all_duplicates = []
        for j in symmetric:
            indices = [i for i, v in enumerate(z) if v == j]
            all_duplicates.append(indices)
        
        # Get a list of lists with the positions of the duplicates
        example_match = matches[0]
        sym_dupl_lists = []
        for sub_list in all_duplicates:
            indices_list = []
            for i in sub_list:
                position = example_match[i]
                indices_list.append(position)
            sym_dupl_lists.append(indices_list)
    
    else:
        sym_dupl_lists = []
    return sym_dupl_lists

# Function to check if an atom has hydrogens attached to it
def has_hydrogens(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetTotalNumHs() > 0



# Function to average shifts for symmetric carbons
def average_shifts(shift_list, sym_groups):
    avg_shifts = {}
    for sym_group in sym_groups:
        avg_shift = [sum([shift_list.get(i, 0)[0] for i in sym_group]) / len(sym_group)]
        for i in sym_group:
            avg_shifts[i] = avg_shift
    return avg_shifts


# Function to update original shift list with averaged values for symmetric carbons
def update_shifts_with_averaged(original_shifts, averaged_shifts):
    updated_shifts = original_shifts.copy()  # Make a copy of the original dictionary
    for carbon, avg_shift in averaged_shifts.items():
        updated_shifts[carbon] = avg_shift  # Update with the averaged value
    return updated_shifts


# Updated function to make plotting optional and return coordinates of plotted points
# Updated function to avoid duplicate peaks in the plot
def plot_and_save_cosy_spectrum_with_zoom_no_duplicates(heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers, plot=True, xlim=None, ylim=None):
    plotted_points = set()  # Using a set to automatically handle duplicates
    
    for carbon1, neighbors in carbon_dict.items():
        h1_shifts = heavy_atom_hydrogen_shift_dict.get(carbon1, [])
        if h1_shifts:
            plotted_points.add((h1_shifts[0], h1_shifts[0]))
            
        for carbon2 in neighbors:
            h2_shifts = heavy_atom_hydrogen_shift_dict.get(carbon2, [])
            if h1_shifts and h2_shifts:
                is_chiral1 = carbon1 in chiral_centers
                is_chiral2 = carbon2 in chiral_centers

                if is_chiral1 or is_chiral2:
                    for h1_shift in h1_shifts:
                        for h2_shift in h2_shifts:
                            plotted_points.add((h1_shift, h2_shift))
                            plotted_points.add((h2_shift, h1_shift))
                else:
                    plotted_points.add((h1_shifts[0], h2_shifts[0]))
                    plotted_points.add((h2_shifts[0], h1_shifts[0]))
                
    x_coords = [x for x, y in plotted_points]
    y_coords = [y for x, y in plotted_points]
    
    if plot:
        plt.scatter(x_coords, y_coords, c='blue', marker='o', label='Cross Peaks', alpha=0.5)
        plt.title('COSY Spectrum with All Diagonal Peaks')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Chemical Shift (ppm)')
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.legend()
        # Apply zooming window if provided
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.show()
    
    return list(plotted_points)  # Converting set back to list




def run_COSY_generation(config):
    folder_path = config.SGNN_gen_folder_path
    nmr_files = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    nmr_files = sorted(nmr_files, reverse=False)

    ### Stores
    smiles_list = []
    data_list = []
    sample_id_list = []

    for file_path in nmr_files[:]:
        try:
            mol = SDMolSupplier(file_path)[0]
            isomers = tuple(EnumerateStereoisomers(mol))
            stereo_smi = Chem.MolToSmiles(isomers[0],isomericSmiles=True)

            # Get NMR shifts
            averaged_nmr_shifts = mol.GetProp('averaged_NMR_shifts')
            sample_shifts = list(map(float, averaged_nmr_shifts.split()))

            # Extract the sample_id from the path
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Find chiral centers
            chiral_centers = find_chiral_centers(mol)

            carbon_dict = find_carbons_with_relevant_neighbors(mol)

            heavy_atom_dict = find_heavy_atoms_with_hydrogens(mol)

            heavy_atom_hydrogen_shift_dict = extract_symmetric_hydrogen_shifts(sample_shifts, heavy_atom_dict)

            sym_dupl_lists = find_symmetric_positions(stereo_smi)

            # Remove symmetric positions that don't have hydrogens
            sym_dupl_lists = [positions for positions in sym_dupl_lists if all(has_hydrogens(mol, idx) for idx in positions)]

            averaged_shifts = average_shifts(heavy_atom_hydrogen_shift_dict, sym_dupl_lists)

            updated_heavy_atom_hydrogen_shift_dict = update_shifts_with_averaged(heavy_atom_hydrogen_shift_dict, averaged_shifts)

            COSY_shifts = plot_and_save_cosy_spectrum_with_zoom_no_duplicates(heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers, plot=False, xlim=None, ylim=None)

            COSY_shifts = sorted(COSY_shifts, key=lambda x: x[0])

            mol = Chem.RemoveHs(mol)
            smi = MolToSmiles(mol)     

            smiles_list.append(smi)
            data_list.append(COSY_shifts)
            sample_id_list.append(sample_id)
        except:
            print(file_path)

    # Create a DataFrame with the lists as columns
    data = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    data.reset_index(drop=True, inplace=True)
    csv_COSY_path = os.path.join(config.SGNN_csv_save_folder, f"data_COSY_{config.ran_num}.csv")
    data.to_csv(csv_COSY_path, index=False)
    return data, csv_COSY_path


##############################################################################
################################ HSQC ########################################
##############################################################################



def run_HSQC_generation(config):
    
    folder_path = config.SGNN_gen_folder_path
    nmr_files = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    nmr_files = sorted(nmr_files, reverse=False)
    ### Stores
    smiles_list = []
    data_list = []
    sample_id_list = []


    for file_path in nmr_files[:]:
        try:
            sample_df = ncfd.load_dft_dft_comparison(file_path)

            # Extract the sample_id from the path
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Use the apply method to create a list of lists
            HSQC_shifts = sample_df.apply(lambda row: [row['F2 (ppm)'], row['F1 (ppm)']], axis=1).tolist()
            HSQC_shifts = sorted(HSQC_shifts, key=lambda x: x[0])

            mol = SDMolSupplier(file_path)[0]   
            mol = Chem.RemoveHs(mol)
            smi = MolToSmiles(mol)   

            smiles_list.append(smi)
            data_list.append(HSQC_shifts)
            sample_id_list.append(sample_id)
        except:
            print(file_path)


    # Create a DataFrame with the lists as columns
    data = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    data.reset_index(drop=True, inplace=True)
    csv_HSQC_path = os.path.join(config.SGNN_csv_save_folder, f"data_HSQC_{config.ran_num}.csv")
    data.to_csv(csv_HSQC_path, index=False)
    return data, csv_HSQC_path

##############################################################################
################################ MAIN ########################################
##############################################################################


def cleanup_files(config):
    """Move NMR SDF files to a dedicated subfolder and clean up other files"""
    # Ensure we have a random number
    if not hasattr(config, 'ran_num'):
        config.ran_num = random.randint(1, 100000)
    
    # Create a subfolder for SDF files
    sdf_subfolder = os.path.join(config.SGNN_gen_folder_path, f"sdf_{config.ran_num}")
    if not os.path.exists(sdf_subfolder):
        os.makedirs(sdf_subfolder, exist_ok=True)
    
    # Find all files in the main folder
    main_folder = config.SGNN_gen_folder_path
    all_files = os.listdir(main_folder)
    
    # Separate files by type
    nmr_sdf_files = []
    non_nmr_sdf_files = []
    mol_files = []
    
    for f in all_files:
        if not os.path.isfile(os.path.join(main_folder, f)):
            continue
            
        if f.endswith('.sdf'):
            if f.startswith('NMR_'):
                nmr_sdf_files.append(f)
            else:
                non_nmr_sdf_files.append(f)
        elif f.endswith('.mol'):
            mol_files.append(f)
    
    print(f"Found {len(nmr_sdf_files)} NMR SDF files to keep")
    print(f"Found {len(non_nmr_sdf_files)} non-NMR SDF files to delete")
    print(f"Found {len(mol_files)} MOL files to delete")
    
    # Move NMR SDF files to the subfolder
    for sdf_file in nmr_sdf_files:
        src_path = os.path.join(main_folder, sdf_file)
        dst_path = os.path.join(sdf_subfolder, sdf_file)
        
        try:
            shutil.copy2(src_path, dst_path)  # Copy with metadata
            os.remove(src_path)  # Remove the original
            # print(f"Moved {sdf_file} to {sdf_subfolder}")
        except Exception as e:
            print(f"Error moving {sdf_file}: {str(e)}")
    
    # Delete non-NMR SDF files
    for sdf_file in non_nmr_sdf_files:
        try:
            os.remove(os.path.join(main_folder, sdf_file))
            # print(f"Deleted non-NMR SDF file: {sdf_file}")
        except Exception as e:
            print(f"Error deleting {sdf_file}: {str(e)}")
    
    # Delete MOL files
    for mol_file in mol_files:
        try:
            os.remove(os.path.join(main_folder, mol_file))
            # print(f"Deleted MOL file: {mol_file}")
        except Exception as e:
            print(f"Error deleting {mol_file}: {str(e)}")
    
    return sdf_subfolder

def main_run_data_generation(config):
    # Ensure necessary directories exist
    if not os.path.exists(config.SGNN_gen_folder_path):
        os.makedirs(config.SGNN_gen_folder_path, exist_ok=True)
    
    if not os.path.exists(config.SGNN_csv_save_folder):
        os.makedirs(config.SGNN_csv_save_folder, exist_ok=True)
    
    # Ensure we have a random number
    if not hasattr(config, 'ran_num'):
        config.ran_num = random.randint(1, 100000)
        print(f"Setting random ran_num: {config.ran_num}")
    else:
        print(f"Using existing ran_num: {config.ran_num}")
    
    # Run the pipeline
    combined_df = run_sgnn(config)
    print("\033[1m\033[33m run_sgnn: DONE\033[0m")
    
    data_1H, csv_1H_path = run_1H_generation(config)
    print("\033[1m\033[33m run_1H_generation: DONE\033[0m")
    print(f"1H CSV saved to: {csv_1H_path}")
    
    data_13C, csv_13C_path = run_13C_generation(config)
    print("\033[1m\033[33m run_13C_generation: DONE\033[0m")
    print(f"13C CSV saved to: {csv_13C_path}")
    
    data_COSY, csv_COSY_path = run_COSY_generation(config)
    print("\033[1m\033[33m run_COSY_generation: DONE\033[0m")
    print(f"COSY CSV saved to: {csv_COSY_path}")
    
    data_HSQC, csv_HSQC_path = run_HSQC_generation(config)
    print("\033[1m\033[33m run_HSQC_generation: DONE\033[0m")
    print(f"HSQC CSV saved to: {csv_HSQC_path}")
    
    # Print data shapes for verification
    print(f"1H data shape: {data_1H.shape}")
    print(f"13C data shape: {data_13C.shape}")
    print(f"COSY data shape: {data_COSY.shape}")
    print(f"HSQC data shape: {data_HSQC.shape}")
    
    # Clean up files at the very end
    print("\033[1m\033[33m Starting final cleanup...\033[0m")
    sdf_subfolder = cleanup_files(config)
    print(f"Organized SDF files into: {sdf_subfolder}")
    print("\033[1m\033[33m Cleanup complete\033[0m")
    
    return combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path



def create_CLIP_dataloaders(config, stoi, stoi_MF):
    #NUM_WORKERS = os.cpu_count()
    batch_size = config.CLIP_batch_size

    train_dataset = dl.MultimodalData(config, 
                                   stoi, 
                                   stoi_MF, 
                                   mode="train")
    
    train_sampler = cf.WeightSortedBatchSampler(train_dataset, 
                                             batch_size, 
                                             drop_last=False)

    test_dataset = dl.MultimodalData(config, 
                                  stoi, 
                                  stoi_MF, 
                                  mode="test")

    test_sampler = cf.WeightSortedBatchSampler(test_dataset, 
                                            batch_size, 
                                            drop_last=False)
    
    val_dataset = dl.MultimodalData(config, 
                                stoi, 
                                stoi_MF, 
                                mode="val")

    #val_sampler = cf.WeightSortedBatchSampler(val_dataset, 
    #                                        1, 
    #                                        drop_last=False)
        
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_sampler=train_sampler,
                                  #shuffle=True, 
                                  collate_fn=dl.collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)
    
    test_dataloader = DataLoader(test_dataset, 
                                  batch_sampler=test_sampler,
                                  #shuffle=False, 
                                  collate_fn=dl.collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1,  # because will run multimodal in parallel on many samples
                                  shuffle=False, 
                                  collate_fn=dl.collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)

    dataloaders = {"train":train_dataloader, "test":test_dataloader, "val":val_dataloader} #, "all":all_dataloader}
    return dataloaders

