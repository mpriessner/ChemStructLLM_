#import plotly.graph_objs as go
from IPython.display import display
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import os
from tqdm import tqdm
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount,EnumerateStereoisomers
from rdkit.Chem import SDMolSupplier, Draw, MolFromSmiles, MolToSmiles
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt

from rdkit.Chem import rdmolfiles
import collections

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


def generate_COSY_dataframe(shifts):
    # save shifts as dataframe
    df_dft = pd.DataFrame()
    c_shifts = np.array(shifts, dtype=object)[:,0] 
    h_shifts = np.array(shifts, dtype=object)[:,1]
    df_dft['F2 (ppm)'] = h_shifts
    df_dft['F1 (ppm)'] = c_shifts
    return df_dft

