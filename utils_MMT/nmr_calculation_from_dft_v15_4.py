import ast
import collections
import copy
import glob
import math
import os
import pandas as pd
import sys
import time
from IPython.display import SVG, display
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles, PandasTools, rdmolfiles, rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, GetStereoisomerCount

# Custom utilities
# sys.path.insert(0, "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/1.1_paper_code")
from utils_MMT.plotting_v15_4 import plot_compare_scatter_plot, plot_compare_scatter_plot_without_direction
from utils_MMT.functions_HSQC_sim_v15_4 import similarity_calculations
from utils_MMT.helper_functions_pl_v15_4 import smile2SDF

# Miscellaneous settings
IPythonConsole.ipython_useSVG = True

# Get the indices of the atoms for calculating the reference shift for carbon and hydrogen
def get_solvent_ref_shifts(solvent_path):
    """ This functions calculates the reference shifts for the solvent and returns the C and H shift"""
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(solvent_path)

    index_list_C =[]
    index_list_H =[]
    start = False

    with open(solvent_path) as f:
        for i in f:

            if "V2000" in i:
                start = True
                counter = -1   #because first round already adds one count without really matchin anything'

            if start:
                if "C" in i:
                    index_list_C.append(counter) 
                if "H" in i:
                    index_list_H.append(counter)
                if "END" in i:
                    break
                counter += 1

    # Get the shielding values from the file
    ticker_shielding = False
    float_shielding_list = []
    with open(solvent_path) as f:
        for idx, i in enumerate(f):
          # get the value streight after the keyword
            if ticker_shielding:
                # To get the same structure as for the Gaussian example save the list in the same way [name, conformer_energy, shifts]
                count = idx               # start the counting of lines from there -> in case the shieldings go over multiple lines
                while ">" not in docline_list[count]:
                    shielding_values = docline_list[count]
                    if shielding_values != "\n":    # This is the separation to the next section
                        float_shielding_list = float_shielding_list + [float(i) for i in shielding_values[:-2].split()]  # convert string to float values and save them in the dict
                    count += 1
                ticker_shielding = False
            if "atom.dprop.Isotropic sheilding" in i:
                ticker_shielding = True

    C_shieldings = []
    H_shieldings = []
    for atom, shielding in zip(atom_list, float_shielding_list):
        if atom == "C":
            C_shieldings.append(shielding)
        if atom == "H":
            H_shieldings.append(shielding)      

    C13_shield_calc = sum(C_shieldings)/len(C_shieldings)
    H1_shield_calc = sum(H_shieldings)/len(H_shieldings)
    return C13_shield_calc, H1_shield_calc


# # Get the indices of the atoms for calculating the reference shift for carbon and hydrogen
# ### ALSO IN NMR SHIFT RECONSTRUCTION
def get_molecule_data(compound_path):
    """ This function returns the list of atoms of the molecules 
    and a list of strings with each line of the compound document as one string 
    because reading the SDF files with pandas causes an error"""
    
    ################ atom_list ################
    index_list_C =[]
    index_list_H =[]
    docline_list = []
    start = False
    name = compound_path.split("/")[-1].split(".")[0]

    # save each string line of the document in a list
    with open(compound_path) as f:
        for i in f:
            docline_list.append(i)

    # get index location of C and H and atom list
    atom_list = []
    stop_list = ["1","0","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
    for i in docline_list:

        if start:
            if "C" in i:
                index_list_C.append(counter) 
            if "H" in i:
                index_list_H.append(counter)
            if (i.split()[0] in stop_list):   # to find the end of the atom defining list
                break
            atom = i.split()[3]
            if atom != "0":
                atom_list.append(atom)
                counter += 1

        if "V2000" in i:   # select when to start looking
            start = True
            counter = 0    
            
    ################ connectivity_list ################
    # # To get the reference right add one empty string to the list -> need to check that there is something wrong
    # atom_list_ = [" "] + atom_list
    atom_list_ = atom_list
    start_line = len(atom_list_)+4
    end_line = len(atom_list_)+4 + len(atom_list_)
    connectivity_list = []

    for idx, i in enumerate(docline_list):
        if idx >= start_line and "M  END" not in i:
            add_list = i.split()
            ### if there are more than 100 connection the first and second columns are connected
            ### therefore I need to manually split them
            if len(add_list) ==3:
                part_1 = str(int(i[:3]))
                part_2 = str(int(i[3:6]))
                part_3 = str(int(i[6:9]))
                part_4 = str(int(i[9:12]))
                add_list =[part_1,part_2,part_3,part_4]

            ### For some reason sometimes it is too long and sometimes it is too short
            if add_list[0]=="M" or add_list[0]==">" or add_list[0]=="$$$$":
                pass
            else:
                connectivity_list.append(add_list)
        if  "M  END" in i:
            break  
    
    ################ mol ################
    # save each string line of the document in a list
    with open(compound_path[:-4]+".mol", "w",  encoding='utf-8', errors='ignore') as output:  # Path to broken file
        with open(compound_path) as f:
            for element in f:
                if "END" in element:
                    output.write(element)
                    break
                else:
                    output.write(element)

    mol = Chem.MolFromMolFile(compound_path[:-4]+".mol")
    return atom_list, connectivity_list, docline_list, name, mol

def get_conformer_energy_list(docline_list):
    """ Returns the list of all the energies fo the different conformers"""
    ticker_shielding = False
    conformer_energy_list = []
    for idx, i in enumerate(docline_list):
        # get the value streight after the keyword
        if ticker_shielding:
            conformer_energy_list.append(float(i))
            ticker_shielding = False
        if "conformer_energy" in i:
            ticker_shielding = True
    return conformer_energy_list


def get_shielding_for_conformers(docline_list, name):
    """ this Function reads out the energies from the provided file (saved in the docline_list) and 
    extracts all the shielding tensors and saves them in from of a list"""
    ticker_shielding = False
    shielding_values_list = []
    conformer = 0
    shielding_tensors_list = []
    float_shielding_list = [name+"_"+str(conformer)] 

    for idx, i in enumerate(docline_list):
        # get the value streight after the keyword
        if ticker_shielding:
            # To get the same structure as for the Gaussian example save the list in the same way [name, conformer_energy, shifts]
            conformer_energy_list = get_conformer_energy_list(docline_list)
            float_shielding_list = float_shielding_list + [conformer_energy_list[conformer]]
            count = idx               # start the counting of lines from there -> in case the shieldings go over multiple lines
            while ">" not in docline_list[count]:
                shielding_values = docline_list[count]
                if shielding_values != "\n":    # This is the separation to the next section
                    float_shielding_list = float_shielding_list + [float(i) for i in shielding_values[:-2].split()]  # convert string to float values and save them in the dict
                count += 1
            shielding_tensors_list.append(float_shielding_list)
            ticker_shielding = False
            conformer += 1
            float_shielding_list = [name+"_"+str(conformer)] 
            
        # find the right line in the document when to start searching for the shielding values
        if "atom.dprop.Isotropic sheilding" in i:  
            ticker_shielding = True
    
    # sort from smallest to biggest scf energy
    shielding_tensors_list.sort(key = lambda x: x[1])
    
    return shielding_tensors_list, conformer_energy_list

def calc_rel_en(shielding_tensors_list):
    """ Calculates the differences in the energies for every conformation"""
    minimum_val = shielding_tensors_list[0][1]
    list_min_val = [0]   # First item is zero
    for i in shielding_tensors_list[1:]:
        list_min_val.append(i[1]-minimum_val)
    return list_min_val



def calculate_boltzmann_factors(list_min_val):
    """ Calculates the boltzman percentage contribution of each conformer"""
    # Variables
    T = 298.15
    k = 0.001987204
    kT = k*T
    
    list_boltzmann_factors = []
    for i in list_min_val:
        val = math.exp(-i/kT)
        list_boltzmann_factors.append(val)

    partition_function_Q = sum(list_boltzmann_factors)
    # boltzmann_perc = list_boltzmann_factors/partition_function_Q
    boltzmann_perc = np.array(list_boltzmann_factors)/partition_function_Q

    return boltzmann_perc, partition_function_Q, list_boltzmann_factors

def perform_boltzmann_averaging(shielding_tensors_list,list_boltzmann_factors,partition_function_Q):
    """ update shielding tensors based on boltzmann averaging """

    master_shielding_list = []
    for idx, shielding_list in enumerate(shielding_tensors_list):
        new_shielding_list = []
        for j in shielding_list[2:]:
            val =  list_boltzmann_factors[idx]/ partition_function_Q *j
            new_shielding_list.append(val)
        master_shielding_list.append(new_shielding_list)
        
    # Add all the contribution together
    boltzman_avg_shifts = [sum(i) for i in zip(*master_shielding_list)]
    return boltzman_avg_shifts


def calculate_nmr_shifts_from_shieldings(atom_list,boltzman_avg_shifts,C13_shift_exp,C13_shield_calc,H1_shield_calc,H1_shift_exp):
    """ This function corrects the carbon and hydrogen shifts based on the standards of 1H and 13C solvent DFT calculation to correct the 
    shift and saved the corrected shifts inplace in boltzman_avg_shifts """
    import copy
    boltzman_avg = copy.deepcopy(boltzman_avg_shifts)
    for idx, (shieldings,atom) in enumerate(zip(boltzman_avg, atom_list)):
        if atom == "C":
            c_shift = C13_shield_calc-shieldings+C13_shift_exp
            boltzman_avg[idx] = c_shift
            
        if atom == "H":
            h_shift = H1_shield_calc-shieldings+H1_shift_exp
            boltzman_avg[idx] = h_shift
    return boltzman_avg

# Parameters from this website: http://cheshirenmr.info/Recommendations.htm
# http://cheshirenmr.info/ScalingFactors.htm#table5dimethylsulfoxideheading
def calculate_nmr_shifts_from_shieldings_slope_intercept(atom_list,boltzman_avg_shifts, slope_H, intercept_H, slope_C, intercept_C):
    """ This function corrects the carbon and hydrogen shifts based on the standards of 1H and 13C solvent DFT calculation to correct the 
    shift and saved the corrected shifts inplace in boltzman_avg_shifts 
    based on: http://cheshirenmr.info/Instructions.htm"""
    import copy
    boltzman_avg = copy.deepcopy(boltzman_avg_shifts)

    for idx, (atom, shielding) in enumerate(zip(atom_list,boltzman_avg)):
        if atom =="C":
            shift_C = (intercept_C-shielding)/-slope_C
            boltzman_avg[idx] = shift_C
        if atom == "H":
            shift_H = (intercept_H-shielding)/-slope_H
            boltzman_avg[idx] = shift_H
    return boltzman_avg

def save_results_sdf_file(conformer_energy_list, docline_list, boltzman_avg_shifts_corr_2, compound_path):
    """ this function saves the final shift predictions into a SDF files with the lowest energy conformer"""
    lowest_conformer_index = conformer_energy_list.index(min(conformer_energy_list))
    
    ### Select the lowest energy conformer
    selected_confirmer = []
    ticker = False
    for i in docline_list:

        if  f"0:0:{lowest_conformer_index}\n" in i:
            ticker = True

        if f"0:0:{lowest_conformer_index+1}\n" in i:
            break

        if ticker == True:
            selected_confirmer.append(i)
    
    ### generate string that will be added in the file (with all the calculated shifts
    string = ""
    for i in boltzman_avg_shifts_corr_2:
        string = string + " " + str(i)
    final_string = string[1:]+"\n"
    final_string
    
    ### add NMR shift data at the end before the name of the compound
    selected_confirmer.insert(-1, f">  <averaged_NMR_shifts>  ({lowest_conformer_index+1}) \n")
    selected_confirmer.insert(-1, final_string)
    selected_confirmer.insert(-1, "\n")

    ### Save file in selected location
    save_path = compound_path.split(".")[0]+"_shifts.sdf"
    with open(os.path.join(save_path), 'w') as f:
        for i in selected_confirmer:
            f.write(i)
    return save_path

# from nmr_calculation_from_dft import save_results_sdf_file, get_solvent_ref_shifts, get_molecule_data, get_shielding_for_conformers, calc_rel_en, calculate_boltzmann_factors, perform_boltzmann_averaging, calculate_nmr_shifts_from_shieldings, calculate_nmr_shifts_from_shieldings_slope_intercept
# Calculate ref values from solvent
def run_script(solvent_path, compound_path, C13_shift_exp, H1_shift_exp, \
               slope_H, intercept_H, slope_C, intercept_C, extra_info=True):
    
    C13_shield_calc, H1_shield_calc = get_solvent_ref_shifts(solvent_path)

    
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(compound_path)

    
    shielding_tensors_list, conformer_energy_list = get_shielding_for_conformers(docline_list, name)


    list_min_val = calc_rel_en(shielding_tensors_list)


    boltzmann_perc, partition_function_Q, list_boltzmann_factors = calculate_boltzmann_factors(list_min_val)


    boltzman_avg_shifts = perform_boltzmann_averaging(shielding_tensors_list,list_boltzmann_factors,partition_function_Q)

    # #DMSO as reference
    # boltzman_avg_shifts_corr_1 = calculate_nmr_shifts_from_shieldings(atom_list,boltzman_avg_shifts,C13_shift_exp,C13_shield_calc,H1_shield_calc,H1_shift_exp)
    # # print(boltzman_avg_shifts_corr_1)

    # TMS as reference
    boltzman_avg_shifts_corr_1 =  calculate_nmr_shifts_from_shieldings_slope_intercept(atom_list,boltzman_avg_shifts, slope_H, intercept_H, slope_C, intercept_C)
    print(boltzman_avg_shifts_corr_1)

    save_path = save_results_sdf_file(conformer_energy_list, docline_list, boltzman_avg_shifts_corr_1, compound_path)
    if extra_info:
        print(C13_shield_calc, H1_shield_calc)
        print("---------------------------------------------")
        print(atom_list, name)
        print("---------------------------------------------")
        print(shielding_tensors_list, conformer_energy_list)
        print("---------------------------------------------")
        print(list_min_val)
        print("---------------------------------------------")
        print(boltzmann_perc,partition_function_Q, list_boltzmann_factors)
        print("---------------------------------------------")
        print(boltzman_avg_shifts)
        print("---------------------------------------------")
        print(boltzman_avg_shifts_corr_1)
        print("---------------------------------------------")    
        # print(boltzman_avg_shifts_corr_2)
        print("---------------------------------------------")
        print(save_path)
        print("---------------------------------------------")
    return save_path, boltzman_avg_shifts_corr_1
# save_path, boltzman_avg_shifts_corr_2 = run_script(solvent_path, compound_path, C13_shift_exp, H1_shift_exp, slope_H, intercept_H, slope_C, intercept_C, extra_info=False)

###############################################################################################

def get_HSQC_info_data(mol=None, smiles="", file_path="", extra_info = False):
    """ This function takes a smiles and calculates the carbons that are connected
    to H that will be displayed in an HSQC spectrum and it returns a dict that
    has the labels of the C with connections to which H
    does not consider chiral compound peak splits yet
    it just takes SMILES or the path to an SDF file
    """
    if mol!=None:
        m = mol_with_atom_index(mol,True)
        am = Chem.GetAdjacencyMatrix(m)
        
    elif smiles != "":
        mol = MolFromSmiles(smiles)
        m = mol_with_atom_index(mol,True)
        am = Chem.GetAdjacencyMatrix(m)
        
    elif file_path!="":
        mol = Chem.MolFromMolFile(file_path)
        m = mol_with_atom_index(mol,True)
        am = Chem.GetAdjacencyMatrix(m)
        
    elements = [(atom.GetSymbol(),idx) for idx, atom in enumerate(m.GetAtoms())]

    if extra_info:
        m = Chem.RemoveAllHs(m)
        smi = MolToSmiles(m)
        m = MolFromSmiles(smi)
        display(m)
        print(elements)
    
    # check for C-H neighbors
    count_list = {}
    for atom, nr in elements:
        if atom == "C":
            for idx, connector in enumerate(am[nr]):

                if connector == 1 and elements[idx][0] =="H":
                    try:
                        # if there is no carbon in the dict yet it will fail and go to except
                        type(count_list[nr]) == list
                        count_list[nr]+=[elements[idx][1]]
                    except:
                        count_list[nr]=[elements[idx][1]]
                        
    # get stereo smile
    # num_iso = GetStereoisomerCount(mol)

    try:
        isomers = tuple(EnumerateStereoisomers(mol))
        stereo_smi = Chem.MolToSmiles(isomers[0],isomericSmiles=True)
    except:
        stereo_smi = Chem.MolToSmiles(mol,isomericSmiles=True)

    return count_list, m, stereo_smi, am, elements


# from utils.functions import mol_with_atom_index
def mol_with_atom_index(mol, include_H=True):
    """ 
    Takes a mol as input and adds H to them
    Visualizes the number which is assigned to which atom as a mol
    if include_H is True then H will be added and also labeled in the mol file
    """
    # mol = MolFromSmiles(smiles)
    if include_H:
        mol = Chem.AddHs(mol, addCoords=True)

    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


### Get the chiral carbons from the compound
def get_chiral_carbons(stereo_smi):
    """ this function gives the chiral carbons based on the stereo-smiles provided and returns the number of the chiral carbon
    based on the occurence of that hetero-atom within the smiles string"""
    chiral_centers = []
    heteroatom_count = 0
    for idx, atom in enumerate(stereo_smi):
        try:
            if (atom =="C" or atom =="c") and stereo_smi[idx+1]=="@":
                chiral_centers.append(heteroatom_count)

            # Catch NH+ compounds that have isomeric centers
            if atom =="N" and stereo_smi[idx+1]=="@":  
                chiral_centers.append(heteroatom_count)

            if atom not in ["H","@","+","-","(",")","[","]","1","2","3","4","5","6","7","8"]:   # just defining the symbols that are not counted in the heteroatom count
                heteroatom_count +=1
        except:
            pass
    return chiral_centers


def check_terminal_chiral_center(chiral_C, am, elements):
    """ This function checks if a given C atom has 3 different distinct neighbors"""
    surrounding_list = []
    for idx, j in enumerate(am[chiral_C]):
        #not refering to itself and no C because otherwise it is not a terminal steroecenter
        if j == 1  and elements[idx][1] != chiral_C and elements[idx][0] != "C": 
            surrounding_list.append(elements[idx][0])
    # print(surrounding_list)
    # check if there are 3 different substituents but no C because otherwise it is not a terminal stereocenter
    if len(set(surrounding_list))==3:
        return True
    else:
        return False  

    
# get ID from direct C neighbor
# and check if it is a terminal chiral center and save it in a separate dict
from collections import defaultdict
def chiral_center_type(chiral_centers, elements, am):
    """This function checks if the chiral center is at the end of a chain (terminal) 
    or somewhere in the middle of the molecule"""
    direct_neighbor_dict = defaultdict(list)
    terminal_neighbor_dict = defaultdict(list)
    direct_neighbor_list = []
    terminal_neighbor_list = []
    # chiral_C = 1   # for now just on one

    for chiral_center in chiral_centers:

        #check if chiral center is terminal
        terminal_neighbor = check_terminal_chiral_center(chiral_center, am, elements)
        if terminal_neighbor:
            for idx,i in enumerate(am[chiral_center]):
                if i == 1 and elements[idx][0]=="C":
                    terminal_neighbor_list.append(idx)
                    terminal_neighbor_dict[chiral_center].append(idx)
        else:
            for idx,i in enumerate(am[chiral_center]):
                if i == 1 and elements[idx][0]=="C":
                    direct_neighbor_list.append(idx)
                    direct_neighbor_dict[chiral_center].append(idx)

    return direct_neighbor_list, direct_neighbor_dict, terminal_neighbor_list, terminal_neighbor_dict


def get_bond_type(idx,ring_bound_list):
    """ As the function name implies"""
    for i in ring_bound_list:
        if idx in i and i[2] == "AROMATIC":
            bond_type = "AROMATIC"
        else:
            bond_type = "SINGLE"
        break
    return bond_type


# check if chiral center is next to aromatic ring then don't split the second layer distance
# because if chiral center is next to aromatic ring the second layer doesn't split
def get_exclude_second_degree_list(mol, chiral_centers, elements, am):
    """This function checks if chiral centers are next to aromatic rings 
    in that case no second layer gets split"""
    ringinfo = mol.GetRingInfo()
    rings = ringinfo.AtomRings()
    ring_atoms = [i for ring in rings for i in ring]
    ring_bound_list = []
    for ring_atoms in rings:
        for i in range(len(ring_atoms)-1):
            bound_type = str(mol.GetBondBetweenAtoms(ring_atoms[i], ring_atoms[i+1]).GetBondType())
            ring_bound_list.append([ring_atoms[i], ring_atoms[i+1], bound_type])

    exclude_second_degree_list = []
    if len(chiral_centers) !=0:
        for i in chiral_centers:
            # print(i)
            for idx, j in enumerate(am[i]):
                if (j == 1) and elements[idx][0] == "C" and idx in ring_atoms:
                    bond_type = get_bond_type(idx,ring_bound_list)
                    if bond_type  == "AROMATIC":
                        exclude_second_degree_list.append(i)
    return exclude_second_degree_list, ring_bound_list



# remove the reference that is from the chiral center that is next to the ring
def get_second_degree_neighbor(direct_neighbor_dict, exclude_second_degree_list, am, elements, direct_neighbor_list, chiral_centers, terminal_neighbor_list):
    """This function removes the chiral centers next to aromatic rings and addes the positions that
    with a distance of two to the list"""
    import copy
    direct_neighbor_dict_ = copy.deepcopy(direct_neighbor_dict)

    if len(direct_neighbor_dict_) !=0:
        for i in exclude_second_degree_list:
            del direct_neighbor_dict_[i]

    # get second degree neighbor
    sec_direct_neighbor = []

    for key,values in direct_neighbor_dict_.items():
        # print(key,values)
        for neighbor in values:
            # print(neighbor)
            for idx, i in enumerate(am[neighbor]):
                if (i == 1 and elements[idx][0]=="C") and (elements[idx][1] not in list(chiral_centers)) and (elements[idx][1] not in direct_neighbor_list) and (idx not in terminal_neighbor_list):
                    direct_neighbor_dict[key].append(idx)
                    sec_direct_neighbor.append(idx)
    return sec_direct_neighbor, direct_neighbor_dict


def get_all_chiral_split_positions(direct_neighbor_dict, terminal_neighbor_dict):
    """As the function says"""
    all_split_positions = []
    for k,v in direct_neighbor_dict.items():
        for i in v:
            all_split_positions.append(i)

    for k,v in terminal_neighbor_dict.items():
        for i in v:
            all_split_positions.append(i)
    return all_split_positions


# Check if those neighbors have symmetric substitutes - if so pop them because will not be duplicated
def check_for_symmetric_substituents(all_split_positions, chiral_centers, elements, am):
    """As the function says"""
    remove_neighbor_list = []
    for i in all_split_positions:
        surrounding_list=[]
        for idx, j in enumerate(am[i]):

            if j == 1  and elements[idx][1] != chiral_centers and elements[idx][1] not in all_split_positions: 
                surrounding_list.append(elements[idx][0])

        ### Remove the neighbor if all substituents are the same or if there are 3 equivalent attoms attached to that carbon (e.g. 3xH)
        if len(set(surrounding_list)) == 1 and len(surrounding_list) ==3:
            index_to_delete = all_split_positions.index(i)
            sym_terminal_neighbor = all_split_positions.pop(index_to_delete)
            
            # to catch the case if it is a ring with 2 C as connection - the I don't want the direct neighbor being removed
            if len(surrounding_list)!= 2:
                remove_neighbor_list.append(sym_terminal_neighbor)
    return remove_neighbor_list


#################### this is a weird exception #########################
# maybe do something about polar compound structures
########################################################################
# # these index carbons will see a split in their spectrum because of the chiral center close by
# # to catch this case: smile_aa = "Fc1c(cccc1CC(Br)(F)OC)CC(O)CC"
# # the second last C in the chain is also not split for some reason in ACD labs?!
# # this has to do with the electronegativity next to the chiral center if polar F or benzo ring then no split
def remove_symmetric_centers(all_split_positions, remove_neighbor_list, sec_direct_neighbor, elements, am):
    """Just if it is e.g. CH3"""
    for i in remove_neighbor_list:
        if (i in sec_direct_neighbor) and (i in all_split_positions):
            for idx, j in enumerate(am[i]):
                if j == 1  and elements[idx][0] =="C" and idx != i:
                    # just delete if there is no other "CH2" spacer
                    # h_num = check_number_of_h(am, jdx)
                    # if h_num ==2:
                    if idx in all_split_positions: #otherwise crashes for some compounds
                        index_to_delete = all_split_positions.index(idx)
                        all_split_positions.pop(index_to_delete)
    return all_split_positions


# check if those carbon positions have 2 hydrogens attached which can be distinguished
# if not pop them as well
# need to do a copy otherwise it doesn't loop over everything
def check_for_two_H(all_split_positions, elements, am):
    """ remove position if there is just one hydrogen connected to the carbon """
    all_split_positions_ = all_split_positions.copy()
    for i in all_split_positions_:
        num_h = 0
        for idx, j in enumerate(am[i]):
            if j == 1  and elements[idx][0] == "H":
                num_h += 1
        if num_h != 2:

            index_to_delete = all_split_positions.index(i)
            all_split_positions.pop(index_to_delete)
    return all_split_positions


def find_symmetric_positions(stereo_smi):
    """https://github.com/rdkit/rdkit/issues/1411"""

    mol = Chem.MolFromSmiles(stereo_smi)
    isomers = tuple(EnumerateStereoisomers(mol))
    stereo_smi_ = Chem.MolToSmiles(isomers[0], isomericSmiles=True, canonical=True)
    mol = Chem.MolFromSmiles(stereo_smi_) 

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



def run_chiral_and_symmetry_finder(smiles="", compound_path="", extra_info=False):

    if compound_path != "":
        mol = Chem.MolFromMolFile(compound_path)
        mol = Chem.AddHs(mol, addCoords=True)
    elif smiles != "":
        mol = MolFromSmiles(smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        compound_path = smile2SDF(smiles)
        
    count_list, m, stereo_smi, am, elements = get_HSQC_info_data(mol=mol, smiles="", file_path="", extra_info = False)
    chiral_centers = get_chiral_carbons(stereo_smi)
    direct_neighbor_list, direct_neighbor_dict, terminal_neighbor_list, terminal_neighbor_dict = chiral_center_type(chiral_centers, elements, am)
    
    exclude_second_degree_list, ring_bound_list = get_exclude_second_degree_list(mol, chiral_centers, elements, am)
    # print(exclude_second_degree_list,  direct_neighbor_dict)
    
    sec_direct_neighbor, direct_neighbor_dict = get_second_degree_neighbor(direct_neighbor_dict, exclude_second_degree_list, am, elements, direct_neighbor_list, chiral_centers, terminal_neighbor_list)
    
    all_split_positions = get_all_chiral_split_positions(direct_neighbor_dict, terminal_neighbor_dict)
    
    remove_neighbor_list = check_for_symmetric_substituents(all_split_positions, chiral_centers, elements, am)
    
    all_split_positions = remove_symmetric_centers(all_split_positions, remove_neighbor_list, sec_direct_neighbor, elements, am)
    
    all_split_positions = check_for_two_H(all_split_positions, elements, am)
    
    sym_dupl_lists = find_symmetric_positions(stereo_smi)

    if extra_info:
        print(count_list, stereo_smi, am, elements)
        print("---------------------------------------------")
        print(chiral_centers)
        print("---------------------------------------------")
        print(direct_neighbor_dict,terminal_neighbor_dict)
        print("---------------------------------------------")
        print(exclude_second_degree_list)
        print("---------------------------------------------")
        print(sec_direct_neighbor, direct_neighbor_dict)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(sym_dupl_lists)
        print("---------------------------------------------")
    return sym_dupl_lists, all_split_positions, mol, compound_path

# sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=compound_path)
# print(sym_dupl_lists, all_split_positions)



# Get C-H connectivity dict with index starting from 0 to correct it to python convention
def get_c_h_connectivity(connectivity_list, atom_list):
    """ This function checks the connectifity list and creates a dictionary with all the 
    carbons that are connected to hydrogens with their labelled numbers"""
    c_h_connectivity_dict = {}
    # print(connectivity_list)
    for i in connectivity_list:
        selected_atom_nr = int(i[0])-1
        selected_connection_nr = int(i[1])-1
        atom = atom_list[selected_atom_nr]
        connection = atom_list[selected_connection_nr]
        num_connection = atom_list[int(i[2])]
        # check atom C-H bonds and add them to dictionary
        if atom =="C" and connection == "H":
            found_H_nr = [selected_connection_nr]
            found_C_nr = selected_atom_nr
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(c_h_connectivity_dict[found_C_nr]) == list
                c_h_connectivity_dict[found_C_nr]+=found_H_nr
            except:
                c_h_connectivity_dict[found_C_nr]=found_H_nr
        # check atom H-C bonds and add them to dictionary
        if atom =="H" and connection == "C":
            found_C_nr = selected_connection_nr
            found_H_nr = [selected_atom_nr]
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(c_h_connectivity_dict[found_C_nr]) == list
                c_h_connectivity_dict[found_C_nr]+=found_H_nr
            except:
                c_h_connectivity_dict[found_C_nr]=found_H_nr
    return c_h_connectivity_dict


def plot_dft_spectrum(shifts, amplifyer=850, transp=0.50):
    """Plots scatter plot from dataframe"""
    import matplotlib.pyplot as plt

    scatter_x_2 = np.array(shifts)[:,1]
    scatter_y_2 = np.array(shifts)[:,0]
    intensity =np.array(shifts)[:,2]
    group = [ '+ve' if i>0 else '-ve' for i in intensity]

    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    match = [True if i == group[0] else False for i in group]

    for g in np.unique(group):
        # print(scatter_x_2[i], scatter_y_2[i],abs(intensity[i])*amplifyer, label=g,  alpha=transp)
        ax.scatter(scatter_x_2[match], scatter_y_2[match], label=g,  alpha=transp)
    ax.legend()
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.title("Plot")
    plt.show()


    # here we calculate the averaged H shifts and match them with the carbon shifts 
# generating the HSQC spectrum including directions of the intensity

def selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2):
    """This functions takes the connectifity list and the symmetry and chirality information and
    reconstructs the shifts including their splittings and returns it as a final list
    including the direction of the peak based on the number of H bound to C"""
    shifts = []
    # iterate over all items of dict
    for cab_pos,hyd_pos_list in c_h_connectivity_dict.items():
        hyd_val = 0
        # average the hydrogen values
        if len(hyd_pos_list)==3 or (len(hyd_pos_list)==2 and cab_pos not in all_split_positions):
            for hyd_pos in hyd_pos_list:
                hyd_val += boltzman_avg_shifts_corr_2[hyd_pos]
            hyd_val_avg =  hyd_val/len(hyd_pos_list)
        else:
            hyd_val_avg = boltzman_avg_shifts_corr_2[hyd_pos_list[0]]
        # For HSQC-DEPT add the directionsSS
        if len(hyd_pos_list)%2 != 0:
            direction = 1
        else:
            direction = -1

        if len(hyd_pos_list)==2 and cab_pos in all_split_positions:
            for hyd_pos in hyd_pos_list:
                hyd_val_avg = boltzman_avg_shifts_corr_2[hyd_pos]
                shifts.append([boltzman_avg_shifts_corr_2[cab_pos],hyd_val_avg, direction, cab_pos, hyd_pos_list])
        else:
            shifts.append([boltzman_avg_shifts_corr_2[cab_pos],hyd_val_avg, direction, cab_pos, hyd_pos_list])
    return shifts


def perform_deduplication_if_symmetric(shifts, sym_dupl_lists):
    """ This function performs deduplication based on the symmetries found in the compound
    and returns the new corrected shifts"""
    #shifts_arr = np.array(shifts)
    shifts_arr = np.array(shifts, dtype=object)
    C_nums = shifts_arr[:,3]
    if len(sym_dupl_lists) != 0:
        new_shifts = []
        c_done = []
        # iterate over the lists of dublicates
        for carbons in sym_dupl_lists:
            carb_sum = 0
            hyd_sum = 0
            carb_pos = []
            hyd_pos = []
            counter = 0
            # iterate over the single carbon positions of the duplicates
            for carb_num in carbons:
                if carb_num in C_nums:
                    # iterate over all shifts in shifts
                    for C,H,direction,c_pos,h_pos in shifts:
                        # check if the carb number of the shift corresponds to the carb dupl list
                        if carb_num == c_pos:
                            carb_sum+=C
                            carb_pos.append(c_pos)
                            hyd_sum+=H
                            hyd_pos.append(h_pos)
                            counter+=1
                            c_done.append(c_pos)
                else:
                    pass
            # that is necessary for compounds with symmetric N 
            # then counter does not go up and leads to zero division error
            if counter!=0:
                carb_avg = carb_sum/counter
                hyd_avg = hyd_sum/counter
                new_shifts.append([carb_avg,hyd_avg, direction, carb_pos, hyd_pos])

        ### perform averaging of the shifts that come occure more often
        grouped_data = {}
        for row in shifts:
            key = row[-2]
            if key not in grouped_data:
                grouped_data[key] = {'count': 0, 'sum': 0, 'row': row}
            grouped_data[key]['count'] += 1
            grouped_data[key]['sum'] += row[1]

        averaged_shifts = []
        for key in grouped_data:
            avg = grouped_data[key]['sum'] / grouped_data[key]['count']
            new_row = grouped_data[key]['row'][:1] + [avg] + grouped_data[key]['row'][2:]
            averaged_shifts.append(new_row)

        # add all other carb peaks that have not been added
        for C,H,direction,c_pos,h_pos in averaged_shifts:
            # check if the carb number of the hift corresponds to the carb dupl list
            if c_pos not in c_done and c_pos in C_nums:
                new_shifts.append([C,H,direction,[c_pos],h_pos])
                c_done.append(c_pos)
    else:
        new_shifts = shifts

    return new_shifts



def generate_dft_dataframe(shifts):
    # save shifts as dataframe
    df_dft = pd.DataFrame()
    c_shifts = np.array(shifts, dtype=object)[:,0]
    h_shifts = np.array(shifts, dtype=object)[:,1]
    directions = np.array(shifts, dtype=object)[:,2]
    atom_index = list(np.array(shifts, dtype=object)[:,3])
    #atom_index = ['_'.join(map(str, sublist)) for sublist in atom_index]
    atom_index = ['_'.join(map(str, sublist if isinstance(sublist, list) else [sublist])) for sublist in atom_index]

    #c_shifts = np.array(shifts)[:,0]
    #h_shifts = np.array(shifts)[:,1]
    #directions = np.array(shifts)[:,2]
    df_dft['F2 (ppm)'] = h_shifts
    df_dft['F1 (ppm)'] = c_shifts
    df_dft["direction"] = directions
    df_dft["atom_index"] = atom_index
    return df_dft
# df_dft = generate_dft_dataframe(shifts)


# sample_id = "HMDB0000159"
# acd_files = glob.glob("/projects/cc/knlr326/5_Downloads/hmdb_data/hmdb/*")


def check_num_peaks_in_acd(df_acd):
    list_strings = []
    for i,j in zip(df_acd["F2 (ppm)"], df_acd["F1 (ppm)"]):
        string = str(i)+str(j)
        list_strings.append(string)
        num_peaks = len(set(list_strings))
    return num_peaks

def load_acd_dataframe(sample_id, acd_files):
    """as the function says"""
    acd_file_path = [i for i in acd_files if sample_id in i ][0]
    data = PandasTools.LoadSDF(acd_file_path)#, smilesName='SMILES',molColName='Mol',includeFingerprints=False)
    data["HSQC_13C-1H"]
    string_data = str(data['HSQC_13C-1H'][0])
    procd = [d.split(';') for d in string_data.split('\n')]
    procd_2 = [i[0].split("\t") for i in procd]
    df_acd = pd.DataFrame(procd_2[1:],columns=procd_2[0])
    direction = [-1 if "<" in i else 1 for i in df_acd["F2 Atom"]]
    df_acd["direction"] = direction
    
    #deduplicate double entrences
    df_acd_dedup = df_acd.drop_duplicates(
      subset = ['F2 (ppm)', "F1 (ppm)"],
      keep = 'last').reset_index(drop = True)
    return df_acd_dedup, acd_file_path
# df_acd, acd_file_path = load_acd_dataframe(sample_id, acd_files)

def load_acd_dataframe_from_file(acd_file_path):
    """as the function says"""
    data = PandasTools.LoadSDF(acd_file_path)#, smilesName='SMILES',molColName='Mol',includeFingerprints=False)
    data["HSQC_13C-1H"]
    string_data = str(data['HSQC_13C-1H'][0])
    procd = [d.split(';') for d in string_data.split('\n')]
    procd_2 = [i[0].split("\t") for i in procd]
    df_acd = pd.DataFrame(procd_2[1:],columns=procd_2[0])
    direction = [-1 if "<" in i else 1 for i in df_acd["F2 Atom"]]
    df_acd["direction"] = direction
    
    #deduplicate double entrences
    df_acd_dedup = df_acd.drop_duplicates(
      subset = ['F2 (ppm)', "F1 (ppm)"],
      keep = 'last').reset_index(drop = True)
    return df_acd_dedup

def load_real_dataframe_from_file(real_file_path):

    df_real = pd.read_csv(real_file_path, sep="\t|\s+")
    df_real = df_real.rename(columns={"F2ppm": "F2 (ppm)", "F1ppm": "F1 (ppm)"})
    return df_real

def load_real_dataframe(sample_id, real_files):
    filter_id = [i for i in real_files if sample_id in i]
    real_file_path = [i for i in filter_id if "two" in i][0]
    df_real = pd.read_csv(real_file_path, sep="\t|\s+")
    df_real = df_real.rename(columns={"F2ppm": "F2 (ppm)", "F1ppm": "F1 (ppm)"})
    return df_real

def load_mnova_dataframe(sample_id, mnova_files):
    
    file_path = [i for i in mnova_files if sample_id in i][0]
    df_mnova = pd.read_csv(file_path, sep="\t|\s+", names=["id", "F2 (ppm)",  "F1 (ppm)", 'Intensity'])

    return df_mnova, file_path

def load_mnova_dataframe_from_file(file_path):
    
    df_mnova = pd.read_csv(file_path, sep="\t|\s+", names=["id", "F2 (ppm)",  "F1 (ppm)", 'Intensity'])
    return df_mnova


def load_sarotti_data(file_path, mode=["exp","calc1","calc2"], nmr_type=["HSQC", "1H", "13C"]):
    """This function loads the HSQC data from the sarotti files and it can choose between experimental results and the calculated ones.
        To get the 1H and 13C still needs to be implemented"""   
    
    data = pd.read_csv(file_path)
    data_load = data[["exp_F2ppm","exp_F1ppm","calc1_F2ppm","calc1_F1ppm","calc2_F2ppm","calc2_F1ppm"]]
    if mode == "exp":
        if nmr_type == "HSQC":
            df = data_load[["exp_F2ppm","exp_F1ppm"]]
            df = df.rename(columns={"exp_F2ppm": "F2 (ppm)", "exp_F1ppm": "F1 (ppm)"})
            df = df.dropna()
            return df
            
    ################# From the website:
    ########## http://cheshirenmr.info/ScalingFactors.htm#table1aheading
    # slope_H = -1.0936
    # intercept_H = 31.8018
    # slope_C = -1.0533
    # intercept_C = 186.5242
    
    if mode == "calc1":
        if nmr_type == "HSQC":
            #### From DP4 File /projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/5_git_repos/DP5/TMSdata 
            ## Gas phase
            slope_H = -1
            intercept_H =  31.6828083333
            slope_C = -1
            intercept_C = 195.8016

            df = data_load[["calc1_F2ppm","calc1_F1ppm"]]
            df = df.dropna()
            df["calc1_F1ppm"] = df["calc1_F1ppm"].apply(lambda shielding: (intercept_C-shielding)/-slope_C)
            df["calc1_F2ppm"] = df["calc1_F2ppm"].apply(lambda shielding: (intercept_H-shielding)/-slope_H)
            df = df.rename(columns={"calc1_F2ppm": "F2 (ppm)", "calc1_F1ppm": "F1 (ppm)"})
            return df
    if mode == "calc2":
        if nmr_type == "HSQC":
                ################ STILL NEED TO CORRECT THE SHIELDING ##################
            #### From DP4 File /projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/5_git_repos/DP5/TMSdata 
            ## Chloroform phase
            slope_H = -1
            intercept_H =  31.668625
            slope_C = -1
            intercept_C = 196.1301
            df = data_load[["calc2_F2ppm","calc2_F1ppm"]]
            df = df.dropna()
            df["calc2_F1ppm"] = df["calc2_F1ppm"].apply(lambda shielding: (intercept_C-shielding)/-slope_C)
            df["calc2_F2ppm"] = df["calc2_F2ppm"].apply(lambda shielding: (intercept_H-shielding)/-slope_H)
            df = df.rename(columns={"calc2_F2ppm": "F2 (ppm)", "calc2_F1ppm": "F1 (ppm)"})
            return df


def plot_spectra_together(df_list, labels, sample_id):
    """
    Plot multiple 2D NMR spectra together, correcting for points that fall together in ACD Labs spectra.

    Parameters:
    -----------
    df_list: list of pandas.DataFrame
        List of dataframes with the spectra to be plotted.
    labels: list of str
        List of labels to be used for each spectrum.
    sample_id: str
        Identifier for the sample.

    Returns:
    --------
    point_numbers: dict
        Dictionary with the number of data points for each spectrum (excluding ACD Labs spectrum).
    """
    if len(df_list)==2:
        [df_1,df_2] = df_list
    if len(df_list)==3:
        [df_1,df_2,df_3] = df_list
    elif len(df_list)==4:
        [df_1,df_2,df_3,df_4] = df_list
    elif len(df_list)==5:
        [df_1,df_2,df_3,df_4,df_5] = df_list
    elif len(df_list)==6:
        [df_1,df_2,df_3,df_4, df_5, df_6] = df_list
    
    if "acd" in labels:
        acd_idx = labels.index("acd")
        df_acd = df_list[acd_idx]
        acd_num_peaks = check_num_peaks_in_acd(df_acd)

    colors = ["green", "red", "blue", "orange", '#17becf', "#9467bd", '#17becf']
    colors = colors[:len(df_list)]
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    
    point_numbers = {}
    for idx, df in enumerate(df_list):
        print(idx)
        scatter_x_2 = np.array(df['F2 (ppm)'].astype(float))
        scatter_y_2 = np.array(df['F1 (ppm)'].astype(float))
        ax.scatter(scatter_x_2, scatter_y_2, label=labels[idx], color=colors[idx],  alpha=0.5)
        if labels[idx] != "acd":
            point_numbers[labels[idx]] = len(scatter_x_2)
        else:
            point_numbers[labels[idx]] = acd_num_peaks


    ax.legend()
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.grid()
    plt.title(sample_id)
    plt.show()
    return point_numbers

    
# check_num_peaks_in_acd(df_acd["F2 (ppm)"], df_acd["F1 (ppm)"])


def get_similarity_comparison_variations(df_1, df_2, sample_id, similarity_type=["euclidean", "cosine_similarity", "pearson_similarity"], error=["sum","avg"], display_img=False, sim_technique=None):
#"min_sum","euc_dist_zero","euc_dist_advanced","hungarian_zero","hungarian_advanced"
    """ This function calculates the similarities of two HSQC dataframes and returns all 4 different
    similarity measures in from of a list [minSum, euclidean, hungarian, adv_hungarian"""
    input_dfs = {}

    # Plot 2 spectra on top of each other
    if display_img:
        try:
            plot_compare_scatter_plot(df_1, df_2, name = sample_id, transp=0.50, style="both",  direction=False)
        except:
            plot_compare_scatter_plot_without_direction(df_1, df_2, name = sample_id, transp=0.50)

    # calculate Cosine Similarity based on spectra
    sim_min_sum_zero, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="min_sum_zero", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_msz = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_msz = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["min_sum_zero"] = [df1_msz, df2_msz]
    
    sim_euc_dist_zero, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="euc_dist_zero", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_edz = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_edz = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["euc_dist_zero"] = [df1_edz, df2_edz]
    
    sim_hung_dist_zero, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="hung_dist_zero", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_hdz = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_hdz = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["hung_dist_zero"] = [df1_hdz, df2_hdz]
    
    sim_min_sum_trunc, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="min_sum_trunc", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_mst = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_mst = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["min_sum_trunc"] = [df1_mst, df2_mst]
    
    sim_euc_dist_trunc, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="euc_dist_trunc", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_edt = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_edt = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["euc_dist_trunc"] = [df1_edt, df2_edt]
    
    sim_hung_dist_trunc, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="hung_dist_trunc", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_hdt = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_hdt = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])    
    input_dfs["hung_dist_trunc"] = [df1_hdt, df2_hdt]
    
    sim_min_sum_nn, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="min_sum_nn", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_msn = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_msn = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["min_sum_nn"] = [df1_msn, df2_msn]
    
    sim_euc_dist_nn, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="euc_dist_nn", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_edn = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_edn = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["euc_dist_nn"] = [df1_edn, df2_edn]
    
    sim_hung_dist_nn, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="hung_dist_nn", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_hdn = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_hdn = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["hung_dist_nn"] = [df1_hdn, df2_hdn]

    result = [sim_min_sum_zero, sim_euc_dist_zero, sim_hung_dist_zero, sim_min_sum_trunc, sim_euc_dist_trunc, sim_hung_dist_trunc, sim_min_sum_nn, sim_euc_dist_nn, sim_hung_dist_nn]
    return result, input_dfs


def load_shifts_from_sdf(sample_id, files):
    """ This functions load the nmr_shifts from the shift-SDF file"""
    file_path = [i for i in files if sample_id in i][0]
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(file_path)
    for idx, line in enumerate(docline_list):
        if ">  <averaged_NMR_shifts>" in line:
            boltzman_avg_shifts_corr_2  = docline_list[idx+1]
            boltzman_avg_shifts_corr_2  = [float(i) for i in boltzman_avg_shifts_corr_2.split()]
            break
    return boltzman_avg_shifts_corr_2, file_path

def load_shifts_from_sdf_file(file_path):
    """ This functions load the nmr_shifts from the shift-SDF file"""
    # file_path = [i for i in files if sample_id in i][0]
    data = PandasTools.LoadSDF(file_path)
    str_shifts = data["averaged_NMR_shifts"].item()
    try:
        boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split(",")]
    except:
        boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split()]
        
    """
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(file_path)
        for idx, line in enumerate(docline_list):
            if ">  <averaged_NMR_shifts>" in line:
                boltzman_avg_shifts_corr_2  = docline_list[idx+1]
                boltzman_avg_shifts_corr_2  = [float(i) for i in boltzman_avg_shifts_corr_2.split()]
                break"""
    return boltzman_avg_shifts_corr_2



def load_ml_dataframe(sample_id, ml_files):
    """
    Load an ML SDF file into a Pandas DataFrame.

    Parameters:
    -----------
    sample_id: str
        The ID of the sample to load.
    ml_files: List[str]
        A list of file paths to the ML SDF files.

    Returns:
    --------
    df_ml: pd.DataFrame
        A Pandas DataFrame containing the ML data.
    """
    
    ml_file_path = [i for i in ml_files if sample_id in i][0]

    data = PandasTools.LoadSDF(ml_file_path)
    str_shifts = data["averaged_NMR_shifts"].item()
    boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split()]

    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=ml_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(ml_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_ml = generate_dft_dataframe(shifts)
    return df_ml


def load_ml_dataframe_from_file(ml_file_path):
    """ Load ML sdf file to dataframe"""
    
    data = PandasTools.LoadSDF(ml_file_path)
    str_shifts = data["averaged_NMR_shifts"].item()
    boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split()]

    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=ml_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(ml_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_ml = generate_dft_dataframe(shifts)
    return df_ml


def load_dft_dft_comparison(dft_file_path):
    
#     files = glob.glob('/projects/cc/knlr326/1_NMR_project/6_Paper_1_analysis/1_HMDB_conformer_analysis/1_HMDB_ICOLOS_DFT/*/*')

#     files = [i for i in files if "shift" in i]
#     files = [i for i in files if not "NMR2_" in i]
#     files = [i for i in files if sample_id in i]
#     files = [i for i in files if not "mol" in i]
#     dft_file_path = [i for i in files if f"CONF_{conf_num}_" in i][0]
    #import IPython; IPython.embed();

    boltzman_avg_shifts_corr_2 = load_shifts_from_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(dft_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_dft = generate_dft_dataframe(shifts)
    dft_num_peaks = len(df_dft)
    return df_dft

def load_sdf_comparison(sdf_file_path):
    
#     files = glob.glob('/projects/cc/knlr326/1_NMR_project/6_Paper_1_analysis/1_HMDB_conformer_analysis/1_HMDB_ICOLOS_DFT/*/*')

#     files = [i for i in files if "shift" in i]
#     files = [i for i in files if not "NMR2_" in i]
#     files = [i for i in files if sample_id in i]
#     files = [i for i in files if not "mol" in i]
#     dft_file_path = [i for i in files if f"CONF_{conf_num}_" in i][0]

    boltzman_avg_shifts_corr_2 = load_shifts_from_sdf_file(sdf_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=sdf_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(sdf_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_sdf = generate_dft_dataframe(shifts)
    return df_sdf

# ##### NOT sure where i used that one
def load_real_dataframe_from_path(real_file_path):
    """ for loading the real hmdb txt files"""
    df_real = pd.read_csv(real_file_path, sep="\t|\s+")
    # df_real = pd.read_csv(real_file_path, sep="\t|\s+", names=["id", "F2 (ppm)",  "F1 (ppm)", 'Intensity'])

    return df_real


def load_real_df_from_file(path_txt):
    """prepares the datafram from the txt file for plotting the real data"""
    try:
        df_real = pd.read_csv(path_txt, sep="\t|\s+")  
    except:
        df_real = pd.read_csv(path_txt, sep="\t")  

    df_real['F2 (ppm)'] = list(df_real['F2ppm'])
    df_real['F1 (ppm)'] = list(df_real['F1ppm'])
    name = path_txt.split("/")[-1][:11]

    return df_real, name


def load_shifts_from_dp_sdf_file(sdf_file_dp):
    data = PandasTools.LoadSDF(sdf_file_dp)
    chemical_shifts = ast.literal_eval(data["averaged_NMR_shifts"][0])
    return chemical_shifts


def load_dft_dp_comparison(dft_file_path):
    """ Load dp sdf file to dataframe"""
    boltzman_avg_shifts_corr_2 = load_shifts_from_dp_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(dft_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_dft = generate_dft_dataframe(shifts)
    dft_num_peaks = len(df_dft)
    return df_dft

def load_dft_dp_dataframe(dft_file_path):
    """Load a DFT (density functional theory) chemical shift file in SD format to a pandas DataFrame.

    Args:
        dft_file_path (str): The path to the DFT file in SD format.

    Returns:
        pandas.DataFrame: A DataFrame containing the chemical shifts and corresponding atomic positions for the DFT file.
    """
    boltzman_avg_shifts_corr_2 = load_shifts_from_dp_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(dft_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_dft = generate_dft_dataframe(shifts)
    dft_num_peaks = len(df_dft)
    return df_dft




def load_all_df_from_id(sample_id, acd_files, real_files, mnova_files, ml_files, dft_files):
    """ This function loads all the df from all 5 datasets if they exist -> from Notebook 6"""
    
    # Get acd df
    acd_file_path = [i for i in acd_files if sample_id in i][0]
    df_acd = load_acd_dataframe_from_file(acd_file_path)
    acd_num_peaks = check_num_peaks_in_acd(df_acd)
    
    # Get real df
    filter_id = [i for i in real_files if sample_id in i]
    real_file_path = [i for i in filter_id if "two" in i][0]
    df_real = load_real_dataframe_from_file(real_file_path)
    real_num_peaks = len(df_real)
    
    # Get mnova df
    mnova_files = [i for i in mnova_files if not "mol" in i]
    file_path_mnova = [i for i in mnova_files if sample_id in i][0]
    df_mnova = load_mnova_dataframe_from_file(file_path_mnova)
    mnova_num_peaks = len(df_mnova)
    
    # Get ML df  
    ml_file_path = [i for i in ml_files if sample_id in i][0]
    df_ml = load_ml_dataframe_from_file(ml_file_path)
    ml_num_peaks = len(df_ml)

    # Get dft df           
    ## Gaussian DP DFT results
    dft_file_path = [i for i in dft_files if sample_id in i][0]    
    df_dft = load_dft_dp_comparison(dft_file_path)
    dft_num_peaks = len(df_dft)

    
    return df_real, df_acd, df_mnova, df_dft, df_ml


def correct_dft_calc_sdf_file(file_path):
    """Corrects an SDF file with DFT-calculated NMR data.
    
    This function opens an SDF file with DFT-calculated NMR data, replaces a specific line in the file, and
    saves the modified file. It also corrects the averaged NMR shifts property if necessary. Finally, it adds
    the H atoms to the SDF file from the corresponding .mol file and saves the modified SDF file under the
    same name.
    
    Parameters:
    -----------
    file_path: str
        The path to the SDF file to be corrected.
        
    Returns:
    --------
    None
    """
    
    ### First make sure that the mol file with H cooridinated is there
    mol = Chem.SDMolSupplier(file_path)[0]

    # Add H atoms to the molecule
    mol_with_h = Chem.AddHs(mol,addCoords=True)

    # Save the molecule to a MOL file with the same name
    mol_filename = file_path.replace('.sdf', '.mol')
    with open(mol_filename, 'w') as f:  
        f.write(Chem.MolToMolBlock(mol_with_h))
    
    # Open the SDF file
    with open(file_path, 'r') as f:
        content = f.read()
    # Replace the desired line
    content = content.replace('>  <atom.dprop.Isotropic sheilding>', '>  <atom_dprop_Isotropic_sheilding>')

    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
        
    data = PandasTools.LoadSDF(file_path)
    if data["averaged_NMR_shifts"][0].startswith("["):
        data["averaged_NMR_shifts"] = data["averaged_NMR_shifts"][0][1:-1]
    try:
        if data["atom_dprop_Isotropic_sheilding"][0].startswith("["):
            data["atom_dprop_Isotropic_sheilding"] = data["atom_dprop_Isotropic_sheilding"][0][1:-1]
    except:
        pass
    rdkit.Chem.PandasTools.WriteSDF(data, file_path, molColName='ROMol', idName=None, properties=list(data.columns), allNumeric=False)

    ########################################################
    ### add the H to the SDF file again from the mol file###
    folder = os.path.dirname(file_path)
    # Load the .sdf file
    for filename in os.listdir(folder):
        if filename.startswith('NMR_SEL_') and filename.endswith('.sdf'):
            # This is an .sdf file starting with "NMR_SEL_"
            sdf_filename = os.path.join(folder, filename)
            mol_filename = os.path.join(folder, filename.replace('.sdf', '.mol'))
    # Load the .sdf file
    with open(sdf_filename, 'rb') as f:
        lines = f.readlines()
        lines = [line.decode('utf-8') for line in lines]
    molblock = ''.join(lines)

    # Load the .mol file and append lines from the .sdf file after "M  END"
    with open(mol_filename, 'r') as f:
        molblock_with_h = f.read()
        
    ## add extra lines to 3D block from Mol files
    for line in lines[lines.index('M  END\n')+1:]:
        molblock_with_h += line
    # save as corrected sdf file under the same name
    with open(sdf_filename, 'w') as f:
        f.write(molblock_with_h)