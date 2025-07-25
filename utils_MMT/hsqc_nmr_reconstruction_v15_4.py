# Standard library imports
import collections
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, rdmolops, rdmolfiles
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, GetStereoisomerCount

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

    # print(direct_neighbor_list)
    # print(terminal_neighbor_list)
    # print(direct_neighbor_dict)
    # print(terminal_neighbor_dict)
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
            if i in direct_neighbor_dict_:
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

    # print(direct_neighbor_list)
    # print(terminal_neighbor_list)
    # print(direct_neighbor_dict)
    # print(terminal_neighbor_dict)
    return direct_neighbor_list, direct_neighbor_dict, terminal_neighbor_list, terminal_neighbor_dict


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
    """ This function performs deduplication based on the symmetries found in the compound and returns the new corrected shifts"""
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
    df_dft['F2 (ppm)'] = h_shifts
    df_dft['F1 (ppm)'] = c_shifts
    df_dft["direction"] = directions
    return df_dft