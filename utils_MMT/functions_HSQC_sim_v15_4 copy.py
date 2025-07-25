# Standard library imports
import os
import random
import shutil
import sys

# Third-party imports for scientific computing
import numpy as np
import scipy.stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist

# Visualization libraries
import matplotlib.pyplot as plt

# RDKit imports for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize.rdMolStandardize import ChargeParent
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# MolVS for molecule standardization
from molvs import Standardizer

# Local path adjustments



def calculate_weight(smiles_str):
    """ used in compare_molformer input smiles string"""
    m = Chem.MolFromSmiles(smiles_str)
    return ExactMolWt(m)

def generate_fingerprint(input_df_or_smiles, output_type = ["list","tensor", "array"], nbits = 512):
    # just calculate Morgan fingerprint of single example
    try:
        sample = input_df_or_smiles['SMILES']
        pattern1 = Chem.MolFromSmiles(sample)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
        fingerprint = list(fp1)
    except:
        pattern1 = Chem.MolFromSmiles(input_df_or_smiles)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
        fingerprint = list(fp1)
        
    if output_type == "tensor":
        fingerprint = torch.Tensor(fingerprint)
    elif output_type == "array":
        fingerprint = np.array(fingerprint, dtype=np.float16)
    return fingerprint

  
def mol2mol_file(mol, folder=None, name=None):
    """ Saves a smile to a given folder with a given name/ID in sdf format
    adding the H is crutial for the ICOLOS workflow
    ir no name is provided then it saves it under a random number"""
    if name==None:
        rand_num = random.randint(0,1000)
        name = "%04d" % (rand_num)
    if folder==None:
        folder= "/projects/cc/knlr326/4_Everything/trash"
        
    # add 3D information to mol
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    save_path = os.path.join(folder,name+".mol")
    writer = Chem.SDWriter(save_path)
    writer.write(mol)
    writer.close()
    return save_path
    

def filter_out_zeros(input_list_1, input_list_2):
    """
    Filters out pairs of elements from input_list_1 and input_list_2 where both
    elements in the pair are close to zero (Close because I modified the numbers slightly because of duplicate entries for some simulation techniques which would otherwise get lost). Returns two new lists containing the
    filtered elements from input_list_1 and input_list_2, respectively.

    Args:
    - input_list_1: A list of pairs of numbers.
    - input_list_2: A list of pairs of numbers.

    Returns:
    - new_list_1: A list containing the elements from input_list_1 that are not
                  close to zero.
    - new_list_2: A list containing the elements from input_list_2 that correspond
                  to the elements in new_list_1 and are not close to zero.
    """
    new_list_1, new_list_2 = [],[]
    for i,j in zip(input_list_1, input_list_2 ):
        if (abs(i[0])+abs(i[1]))<0.0001 or (abs(j[0]) + abs(j[1]))<0.0001:
            continue
        else:
            new_list_1.append(list(i))
            new_list_2.append(list(j))
    new_list_1 = np.array(new_list_1)
    new_list_2 = np.array(new_list_2)
    return new_list_1, new_list_2


#from utils.functions_1 import euclidean_distance_zero_padded, #euclidean_distance_advanced,min_sum_adv,hungarian_zero_padded,hungarian_advanced_euc,plot_assignment_points
def similarity_calculations(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
                            similarity_type=["euclidean","cosine_similarity","pearson_similarity"],  \
                            error=["sum","avg", "avg_scaled"], \
                            assignment_plot = True, sim_technique=None):
    """This function calculates the cosine similarity of xy of two spectra provided by dataframes
    by first normalizing it and then choosing one of two modes
    min_sum: takes the minimum sum of x + y as a sorting criteria
    euc_dist: compares every point with each other point of the spectra and matches them to minimize the 
    error"""

    h_dim_1 = list(np.array(df_1['F2 (ppm)'].astype(float))/11-0.5)
    c_dim_1 = list(np.array(df_1['F1 (ppm)'].astype(float))/180-0.5)
    
    h_dim_2 = list(np.array(df_2['F2 (ppm)'].astype(float))/11-0.5)
    c_dim_2 = list(np.array(df_2['F1 (ppm)'].astype(float))/180-0.5)

    input_list_1 = [h_dim_1, c_dim_1]
    input_list_2 = [h_dim_2, c_dim_2]
    
    # convert it into the right shape
    input_list_1 = np.array(input_list_1).transpose()
    input_list_2 = np.array(input_list_2).transpose()
    
    #sorting by sum of x,y high to low
    ######### Do alignment algorithm instead

    if mode == "min_sum_zero":
        input_list_1,input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1 = np.array(sorted(input_list_1, key = lambda x: -(x[0]+x[1])))
        input_list_2 = np.array(sorted(input_list_2, key = lambda x: -(x[0]+x[1])))

    elif mode == "min_sum_trunc":
        min_len = min(len(input_list_1), len(input_list_2))
        input_list_1 = np.array(sorted(input_list_1, key = lambda x: -(x[0]+x[1])))[:min_len]
        input_list_2 = np.array(sorted(input_list_2, key = lambda x: -(x[0]+x[1])))[:min_len]
    
    elif mode == "min_sum_nn":
        input_list_1, input_list_2 = min_sum_adv(input_list_1,input_list_2, sim_technique=sim_technique)
        
    elif mode == "euc_dist_zero": 
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num, sim_technique=sim_technique)
        
    elif mode == "euc_dist_trunc": 
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num, sim_technique=sim_technique)
        input_list_1, input_list_2 = filter_out_zeros(input_list_1, input_list_2)     
        
    elif mode == "euc_dist_nn": 
        input_list_1, input_list_2 = euclidean_distance_advanced(input_list_1,input_list_2)

    elif mode == "hung_dist_zero":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)
        
    elif mode == "hung_dist_trunc":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num, sim_technique=sim_technique)
        input_list_1, input_list_2 = filter_out_zeros(input_list_1, input_list_2)     
        input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)
   
    elif mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2 = hungarian_advanced_euc(input_list_1,input_list_2)
    
    
    if assignment_plot == True:
        plot_assignment_points(input_list_1, input_list_2, mode)
        pass
    #import IPython; IPython.embed();    
    
    from scipy.spatial import distance
    if similarity_type == "cosine_similarity":
        from scipy.spatial import distance
        list_points_1 = np.array(input_list_1, dtype=object)
        list_points_2 = np.array(input_list_2, dtype=object)
        Aflat = np.hstack(list_points_1)
        Bflat = np.hstack(list_points_2)
        # Aflat = Aflat - Aflat.mean()
        # Bflat = Bflat - Bflat.mean()
        cos_sim = 1 - distance.cosine(Aflat, Bflat)
        return cos_sim, np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()
    
    elif similarity_type == "euclidean":
        sum_dist = 0
        max_dist = 0
        for sample_1, sample_2 in zip(input_list_1, input_list_2):
            dst = distance.euclidean(sample_1, sample_2)
            sum_dist+=dst
            max_dist = max(dst, max_dist)
            
        if error=="avg":
            ############# new addition #############
            if not "trunc" in mode:
                avg_dist = sum_dist/max(len(input_list_1),len(input_list_2))
            elif  "trunc" in mode:
                avg_dist = sum_dist/min(len(input_list_1),len(input_list_2))    
            return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()
        
        elif error=="sum":
            return np.array(sum_dist), np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()  
        
        elif error=="avg_scaled":
            avg_dist=sum_dist/len(input_list_1)
            avg_dist_scaled = avg_dist/max_dist
            return avg_dist_scaled, np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()  
        
    elif similarity_type == "pearson_similarity":
        pearson_sim = calculate_pearson_similarity(input_list_1, input_list_2)
        return pearson_sim, np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()

    

def calculate_pearson_similarity(spectrum1, spectrum2):
    """Computes the Pearson correlation similarity between two 2D HSQC spectra.

    Parameters:
    spectrum1 (numpy.ndarray): The first spectrum, with shape (n, 2).
    spectrum2 (numpy.ndarray): The second spectrum, with shape (m, 2).

    Returns:
    float: The Pearson correlation similarity between the two spectra.
    """
    # Extract the x and y coordinates from the spectra
    x1 = spectrum1[:, 0]
    y1 = spectrum1[:, 1]
    x2 = spectrum2[:, 0]
    y2 = spectrum2[:, 1]

    # Compute the Pearson correlation coefficient
    correlation, _ = scipy.stats.pearsonr(x1, x2)
    correlation += scipy.stats.pearsonr(y1, y2)[0]
    correlation /= 2
    return correlation


def padding_to_max_length(input_list_1,input_list_2,  dim = 2):
    """ Takes in a nparray of both items
        Perform padding to the longer list"""
    
    if len(input_list_2) > len(input_list_1):
        pad_num = len(input_list_2)-len(input_list_1)
        padding_matrix = np.zeros((pad_num,dim))
        input_list_1 = np.concatenate((input_list_1, padding_matrix), axis=0)
    elif len(input_list_2) < len(input_list_1):
        pad_num = len(input_list_1)-len(input_list_2)
        padding_matrix = np.zeros((pad_num,dim))
        input_list_2 = np.concatenate((input_list_2, padding_matrix), axis=0)
    else:
        pad_num=0
        pass
    return input_list_1,input_list_2, pad_num
    


def plot_assignment_points(input_list_orig, input_list_sim, mode):
    """This function takes the sorted orig and sim inputs and plots them with assignment lines for the point matching algorithms"""
    assignment = list(range(len(input_list_orig)))

    points1 = input_list_orig
    points2 = input_list_sim

    # fig, ax = plt.subplots(figsize=(5,5), dpi=80)

    plt.plot(points1[:,0], points1[:,1],'bo', markersize = 10)
    plt.plot(points2[:,0], points2[:,1],'rs',  markersize = 7)
    for p in range(len(points1)):
        plt.plot([points1[p,0], points2[assignment[p],0]], [points1[p,1], points2[assignment[p],1]], 'k')

    plt.title(mode)
    # plt.axes().set_aspect('equal')
    plt.show()

#######################################################################################################


def min_sum_adv(input_list_1,input_list_2, sim_technique=None):
    """ This function uses min sum for matching the first  points that are up to the same length of t the two 
    sets and then euclidean distance for the remaining points that will be matched twice"""
    input_list_1 = np.array(sorted(input_list_1, key = lambda x: -(x[0]+x[1])))
    input_list_2 = np.array(sorted(input_list_2, key = lambda x: -(x[0]+x[1])))
    min_length = min(len(input_list_1),len(input_list_2))
    input_list_1_part_1 = input_list_1[:min_length]
    input_list_2_part_1 = input_list_2[:min_length]
    input_list_1_align = list(input_list_1_part_1)
    input_list_2_align = list(input_list_2_part_1)
    
    if len(input_list_2) > len(input_list_1):
        input_list_2_part_2 = input_list_2[min_length:]
        
        input_list_1_part_2, input_list_2_part_2  = euclidean_distance_uneven(input_list_1, input_list_2_part_2, sim_technique=sim_technique)
        input_list_1_align.extend(list(input_list_1_part_2))
        input_list_2_align.extend(list(input_list_2_part_2))        

        
    elif len(input_list_2) < len(input_list_1):
        input_list_1_part_2 = input_list_1[min_length:]
        
        input_list_1_part_2, input_list_2_part_2 = euclidean_distance_uneven(input_list_1_part_2, input_list_2, sim_technique=sim_technique)
        input_list_1_align.extend(list(input_list_1_part_2))
        input_list_2_align.extend(list(input_list_2_part_2))     


    return np.array(input_list_1_align), np.array(input_list_2_align)


def hungarian_zero_padded(input_list_1,input_list_2):
    """ From https://stackoverflow.com/questions/39016821/minimize-total-distance-between-two-sets-of-points-in-python
    """
    C = cdist(input_list_1, input_list_2)
    b, assigment = linear_sum_assignment(C)

    # make array to list
    input_list_1_list = [list(i) for i in list(input_list_1)]
    input_list_2_list = [list(i) for i in list(input_list_2)]

    assigned_list = [[input_list_1_list[i],input_list_2_list[b]] for i, b in enumerate(assigment)]
    output_list_1 = [assigned_list[i][0] for i in range(len(assigned_list))]
    output_list_2 = [assigned_list[i][1] for i in range(len(assigned_list))]
    
    output_array_1 = np.array(output_list_1)
    output_array_2 = np.array(output_list_2)
    return np.array(output_array_1), np.array(output_array_2)



def hungarian_advanced_euc(input_list_1,input_list_2):
    #Hungarian advanced
    input_list_1_euc,input_list_2_euc = euclidean_distance_advanced(input_list_1,input_list_2)
    input_list_1_euc_hung,input_list_2_euc_hung = hungarian_zero_padded(input_list_1_euc,input_list_2_euc)
    return np.array(input_list_1_euc_hung),np.array(input_list_2_euc_hung)
   
    
#######################################################################################################
    
def euclidean_distance_zero_padded(input_list_1,input_list_2, num_pad, sim_technique=None):
    """This function aligns the closest points with each other based on euclidean distance
    and matches the remaining ones with the zero padding"""
    
    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
    if sim_technique=="ACD":
        x = []
        y = []
        for i,j in input_list_1:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)

        input_list_1 = np.array([x,y]).transpose()

        x = []
        y = []
        # print(input_list_sim)
        for i,j in input_list_2:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_2 = np.array([x,y]).transpose()    
    ###########################################################################
    
    # calculate euclidean distance
    result = []
    for j in input_list_1:
        for i in input_list_2:
            dst = distance.euclidean(i, j)
            result.append([list(i),list(j),dst])
    
    # sort it for the lowest euclidean distance
    result = sorted(result, key=lambda x:x[-1], reverse=False)

    # This aligns the closest points with each other 
    # and compares the remaining with the zero padding
    dataset_1 = []
    dataset_2 = []
    ### Here I check if any of the combinations was already seen before
    ### Also the matches with the zero padding
    count = 0
    for i in result:
        if ((i[0] not in dataset_2) & (i[1] not in dataset_1)):
            dataset_2.append(i[0])
            dataset_1.append(i[1])
        # ### need to do that count because otherwise zero get more often assigned
        # elif (i[0] == [0,0]) and (i[1] not in dataset_1) and count > num_pad: 
        #     dataset_1.append(i[1])
        #     dataset_2.append([0,0])
        #     count+=1
        # elif (i[0] not in dataset_2) and (i[1] == [0,0]) and count > num_pad:
        #     dataset_2.append(i[0])
        #     dataset_1.append([0,0])
            # count+=1

    return np.array(dataset_1), np.array(dataset_2)
#### To really get the minimum it should minimize on all possible combinations to get the lowest error


def euclidean_distance_uneven(input_list_1,input_list_2, sim_technique=None):
    """This function aligns the closest points with each other based on euclidean distance
    and matches the remaining ones with the zero padding"""
    
    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
    if sim_technique in ["ACD", "ML", "DFT", "DFT1", "DFT2"]:
        x = []
        y = []
        for i,j in input_list_1:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_1 = np.array([x,y]).transpose()

        x = []
        y = []
        # print(input_list_sim)
        for i,j in input_list_2:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_2 = np.array([x,y]).transpose()    
    ###########################################################################
    
    # calculate euclidean distance
    result = []
    for j in input_list_1:
        for i in input_list_2:
            dst = distance.euclidean(i, j)
            result.append([list(i),list(j),dst])
    
    # sort it for the lowest euclidean distance
    result = sorted(result, key=lambda x:x[-1], reverse=False)
    
    # This aligns the closest points with each other 
    # and compares the remaining with the zero padding
    dataset_1 = []
    dataset_2 = []
    ### Here I check if any of the combinations was already seen before
    ### Also the matches with the zero padding
    for i in result:
        # print(i)
        if ((i[0] not in dataset_2) & (i[1] not in dataset_1)):
            dataset_2.append(i[0])
            dataset_1.append(i[1])
    return np.array(dataset_1), np.array(dataset_2)

def euclidean_distance_advanced(input_list_1,input_list_2, sim_technique=None):
    """The Euclidean-Distance-Advanced first matches each point from one set to the second and then picks the 
    remaining points that were not matched because of a mismatch of number of points and matches them again 
    to the shorter set. """
    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
    if sim_technique in ["ACD", "ML", "DFT", "DFT1", "DFT2"]:

        x = []
        y = []
        for i,j in input_list_1:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_1 = np.array([x,y]).transpose()

        x = []
        y = []
        # print(input_list_sim)
        for i,j in input_list_2:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_2 = np.array([x,y]).transpose()   


    ### first alignment of points with euclidean distance with different number of points in each set
    # input_list_1_pad, input_list_2_pad= padding_to_max_length(input_list_1,input_list_2)
    input_list_1_align, input_list_2_align = euclidean_distance_uneven(input_list_1, input_list_2, sim_technique=sim_technique)

    # # Select the remaining points that have not been matched in the first round for second alignment
    input_list_1_align_part_2 = []
    input_list_2_align_part_2 = []            
    if len(input_list_1)< len(input_list_2):
        for i in input_list_2 :
            if i not in input_list_2_align:
                input_list_2_align_part_2.append(i)

        # match them again with full number of points from other set
        input_list_1_align_part_2, input_list_2_align_part_2 = euclidean_distance_uneven(input_list_1,input_list_2_align_part_2, sim_technique=sim_technique)

    elif len(input_list_1)> len(input_list_2):
        for i in input_list_1:
            if i not in input_list_1_align:
                input_list_1_align_part_2.append(i)

        # match them again with full number of points from other set
        input_list_1_align_part_2, input_list_2_align_part_2 = euclidean_distance_uneven(input_list_1_align_part_2,input_list_2, sim_technique=sim_technique)

    # Combine both to final list
    input_list_1_align = list(input_list_1_align)
    input_list_2_align = list(input_list_2_align)
    input_list_1_align.extend(list(input_list_1_align_part_2))
    input_list_2_align.extend(list(input_list_2_align_part_2))
    return np.array(input_list_1_align), np.array(input_list_2_align)


def plot_all_similarities(min_sum_1, min_sum_2, \
                          min_sum_adv_1, min_sum_adv_2,\
                          euc_dist_zero_1, euc_dist_zero_2,\
                          euc_dist_advanced_1, euc_dist_advanced_2,\
                          hungarian_zero_1, hungarian_zero_2,\
                          hungarian_advanced_1, hungarian_advanced_2,\
                          similarities, label, value):
    """ from Notbook 6 - this plots all the comparisons with point assignments to analyse the misassignments. """
    assigment = np.array(range(len(min_sum_1)))
    fig, axs = plt.subplots(3, 2)

    fig.set_figheight(20)
    fig.set_figwidth(15)

    axs[0, 0].plot(min_sum_1[:,0], min_sum_1[:,1],'bo', markersize = 10, label=label[0])
    axs[0, 0].plot(min_sum_2[:,0], min_sum_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[0, 0].plot([5.5], [90.0],'gs',  markersize = 10, label="extra point")
    axs[0, 0].legend()
    N=len(min_sum_1)
    for p in range(N):
        axs[0, 0].plot([min_sum_1[p,0], min_sum_2[assigment[p],0]], [min_sum_1[p,1], min_sum_2[assigment[p],1]], 'k')
    axs[0, 0].set_title(f'{value} - MinSum-Zero: {similarities[0]}')
    axs[0, 0].invert_xaxis()
    axs[0, 0].invert_yaxis()

    axs[0, 1].plot(min_sum_adv_1[:,0], min_sum_adv_1[:,1],'bo', markersize = 10, label=label[0])
    axs[0, 1].plot(min_sum_adv_2[:,0], min_sum_adv_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[0, 1].legend()
    N=len(min_sum_adv_1)
    for p in range(N):
        axs[0, 1].plot([min_sum_adv_1[p,0], min_sum_adv_2[assigment[p],0]], [min_sum_adv_1[p,1], min_sum_adv_2[assigment[p],1]], 'k')
    axs[0, 1].set_title(f'{value} - MinSum-NN: {similarities[1]}')
    axs[0, 1].invert_xaxis()
    axs[0, 1].invert_yaxis()

    axs[1, 0].plot(euc_dist_zero_1[:,0], euc_dist_zero_1[:,1],'bo', markersize = 10, label=label[0])
    axs[1, 0].plot(euc_dist_zero_2[:,0], euc_dist_zero_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[1, 0].plot([5.5], [90.0],'gs',  markersize = 10, label="extra point")
    axs[1, 0].legend()
    N=len(euc_dist_zero_1)
    for p in range(N):
        axs[1, 0].plot([euc_dist_zero_1[p,0], euc_dist_zero_2[assigment[p],0]], [euc_dist_zero_1[p,1], euc_dist_zero_2[assigment[p],1]], 'k')
    axs[1, 0].set_title(f"{value} - EucDist-Zero: {similarities[2]}")
    axs[1, 0].invert_xaxis()
    axs[1, 0].invert_yaxis()
    
    axs[1, 1].plot(euc_dist_advanced_1[:,0], euc_dist_advanced_1[:,1],'bo', markersize = 10, label=label[0])
    axs[1, 1].plot(euc_dist_advanced_2[:,0], euc_dist_advanced_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[1, 1].legend()
    N=len(euc_dist_advanced_1)
    for p in range(N):
        axs[1, 1].plot([euc_dist_advanced_1[p,0], euc_dist_advanced_2[assigment[p],0]], [euc_dist_advanced_1[p,1], euc_dist_advanced_2[assigment[p],1]], 'k')
    axs[1, 1].set_title(f'{value} - EucDist-NN: {similarities[3]}')
    axs[1, 1].invert_xaxis()
    axs[1, 1].invert_yaxis()
    
    axs[2, 0].plot(hungarian_zero_1[:,0], hungarian_zero_1[:,1],'bo', markersize = 10, label=label[0])
    axs[2, 0].plot(hungarian_zero_2[:,0], hungarian_zero_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[2, 0].plot([5.5], [90.0],'gs',  markersize = 10, label="extra point")
    axs[2, 0].legend()
    N=len(hungarian_zero_1)
    for p in range(N):
        axs[2, 0].plot([hungarian_zero_1[p,0], hungarian_zero_2[assigment[p],0]], [hungarian_zero_1[p,1], hungarian_zero_2[assigment[p],1]], 'k')
    axs[2, 0].set_title(f'{value} - HungDist-Zero: {similarities[4]}')
    axs[2, 0].invert_xaxis()
    axs[2, 0].invert_yaxis()
    
    axs[2, 1].plot(hungarian_advanced_1[:,0], hungarian_advanced_1[:,1],'bo', markersize = 10, label=label[0])
    axs[2, 1].plot(hungarian_advanced_2[:,0], hungarian_advanced_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[2, 1].legend()
    N=len(hungarian_advanced_1)
    for p in range(N):
        axs[2, 1].plot([hungarian_advanced_1[p,0], hungarian_advanced_2[assigment[p],0]], [hungarian_advanced_1[p,1], hungarian_advanced_2[assigment[p],1]], 'k')
    axs[2, 1].set_title(f'{value} - HungDist-NN: {similarities[5]}')
    axs[2, 1].invert_xaxis()
    axs[2, 1].invert_yaxis()
    
    for ax in axs.flat:
        ax.set(xlabel='1H-ppm', ylabel='13C-ppm')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    

    
def calculate_tanimoto_from_two_smiles(smi1, smi2, nbits, extra_info = True):
    """This function takes two smile_stings and 
    calculates the Tanimoto similarity and returns it and prints it out"""
    
    pattern1 = Chem.MolFromSmiles(smi1)
    pattern2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(pattern2, 2, nBits=nbits)

    tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    tan_sim = round(tan_sim,4)
    if extra_info:
        print(f"Smile 1: {selection_list[0]} \nSmile 2: {selection_list[1]} \nTanimoto score:{tan_sim}")
    
    ids_compared = selection_list[0]+"_"+selection_list[1]
    return (ids_compared,tan_sim)
    
    
def plot_correlation_plot_conf(all_H_data_non_zero_T, all_C_data_non_zero_T, conf_num, mode):
    """ This function plots a correlation plot of H and C comparing real to simulated datapoints"""
    plt.scatter(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], 1)
    plt.plot(all_H_data_non_zero_T[0], m*all_H_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"H correlation between real and {conf_num} generated spectrum with {mode} alignment")
    #calculate R value
    r_H = np.corrcoef(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1])
    plt.text(0.5,9, f'$R^2$ value: {str(r_H[0][1])[:6]}', fontsize=10)
    #calculate MAE value         
    MAE_H = np.mean(np.absolute(all_H_data_non_zero_T[0]-all_H_data_non_zero_T[1]))
    plt.text(0.5,8.5, f'MAE value: {str(MAE_H)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 11,2))
    plt.yticks(np.arange(0, 11,2))
    plt.show()

    plt.scatter(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], 1)
    plt.plot(all_C_data_non_zero_T[0], m*all_C_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"C correlation between real and {conf_num} generated spectrum with {mode} alignment")
    #calculate R value
    r_C = np.corrcoef(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1])
    plt.text(5,180, f'$R^2$ value: {str(r_C[0][1])[:6]}', fontsize=10)
    #calculate MAE value         
    MAE_C = np.mean(np.absolute(all_C_data_non_zero_T[0]-all_C_data_non_zero_T[1]))
    plt.text(5,170, f'MAE value: {str(MAE_C)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 201,25))
    plt.yticks(np.arange(0, 201,25))
    plt.show()
    
    
def plot_correlation_plot(all_H_data_non_zero_T, all_C_data_non_zero_T, sim_tech, mode):
    """ This function plots a correlation plot of H and C comparing real to simulated datapoints"""
    plt.scatter(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], 1)
    plt.plot(all_H_data_non_zero_T[0], m*all_H_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"H correlation between real and {sim_tech} generated spectrum with {mode} alignment")
    #calculate R value
    r_H = np.corrcoef(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1])
    plt.text(0.5,9, f'$R^2$ value: {str(r_H[0][1])[:6]}', fontsize=10)
    #calculate MAE value         
    MAE_H = np.mean(np.absolute(all_H_data_non_zero_T[0]-all_H_data_non_zero_T[1]))
    plt.text(0.5,8.5, f'MAE value: {str(MAE_H)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 11,2))
    plt.yticks(np.arange(0, 11,2))
    plt.show()

    plt.scatter(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], 1)
    plt.plot(all_C_data_non_zero_T[0], m*all_C_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"C correlation between real and {sim_tech} generated spectrum with {mode} alignment")
    r_C = np.corrcoef(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1])
    plt.text(5,180, f'$R^2$ value: {str(r_C[0][1])[:6]}', fontsize=10)

    #calculate MAE value         
    MAE_C = np.mean(np.absolute(all_C_data_non_zero_T[0]-all_C_data_non_zero_T[1]))
    plt.text(5,170, f'MAE value: {str(MAE_C)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 201,25))
    plt.yticks(np.arange(0, 201,25))
    plt.show()
    return MAE_H, MAE_C


    
def plot_compare_scatter_plot_without_direction(df_orig, df_sim, name="Plot", transp=0.50):
    """Plots scatter plot from two dataframes on top of each other.
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    scatter_x_1 = list(np.array(df_orig['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_orig['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label="Real",  alpha=transp, color="blue")

    scatter_x_2 = list(np.array(df_sim['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_sim['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label="Simulated",  alpha=transp, color="red")

    ax.legend()
    plt.title(name)
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()

    
def plot_compare_scatter_plot_without_direction_all(df_real, df_acd, df_mnova, df_dft, df_ml, labels, transp=0.50):
    """Plots scatter plot from all simulation techniques on top of each other.
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    scatter_x_1 = list(np.array(df_real['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_real['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label=labels[0],  alpha=1, color="green", s=40, marker="s")

    scatter_x_2 = list(np.array(df_acd['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_acd['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[1],  alpha=transp, color="blue")
    
    scatter_x_2 = list(np.array(df_mnova['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_mnova['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[2],  alpha=transp, color="red")
    
    scatter_x_2 = list(np.array(df_dft['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_dft['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[3],  alpha=transp, color="orange")
    
    scatter_x_2 = list(np.array(df_ml['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_ml['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[4],  alpha=transp, color="m")
    
    ax.legend()
    plt.title("All spectra together")
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()

    
     
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
    mol = data.ROMol.item()
    mol = Chem.AddHs(mol, addCoords=True)
    data.ROMol = mol
    rdkit.Chem.PandasTools.WriteSDF(data, file_path, molColName='ROMol', idName=None, properties=list(data.columns), allNumeric=False)


### TEST ###
#from utils.nmr_calculation_from_dft import run_chiral_and_symmetry_finder, get_molecule_data, get_c_h_connectivity, selecting_shifts, load_shifts_from_sdf
#from utils.nmr_calculation_from_dft import load_real_dataframe, get_similarity_comparison_variations, load_shifts_from_sdf_file
#from utils.nmr_calculation_from_dft import perform_deduplication_if_symmetric, generate_dft_dataframe, plot_spectra_together, load_acd_dataframe_from_file
#from utils.nmr_calculation_from_dft import check_num_peaks_in_acd, check_num_peaks_in_acd, load_mnova_dataframe, load_acd_dataframe, plot_spectra_together
#source_folder = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/12_HMDB/HMDB_DFT_ICOLOS_gaus_10_2"
#dft_files = glob.glob(source_folder+"/*/*/*")
#dft_files = [i for i in dft_files if ".sdf" in i]
#dft_files = [i for i in dft_files if "conf" in i]
#len(dft_files)
##dft_files = [i for i in dft_files if "NMR_H2O" in i]
#len(dft_files)
#### Perform corrections of the dft nmr files
#for file_path in tqdm(dft_files[:]):
#    print(file_path)
#    correct_dft_calc_sdf_file(file_path)
    
    


def copy_specific_files(src, dst):
    """Parameters:

    src: str - The path to the source directory.
    dst: str - The path to the destination directory.
    The function creates the destination directory if it doesn't exist. Then it iterates through the subfolders and files in the source directory. If the item is a subfolder, it calls this function recursively. If the item is a file and matches the desired filename pattern (starting with "HMDB" or starting with "SEL_" and ending with ".sdf"), it copies the file to the corresponding location in the destination directory.
    """
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Iterate through the subfolders and files in the source directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)

        # If the item is a subfolder, call this function recursively
        if os.path.isdir(src_item):
            # Skip the subfolder if its name is in the exclusion list
            if item not in ["nmr", "e", "opt", "conf"]:
                copy_specific_files(src_item, dst_item)

        # If the item is a file and matches the desired filename pattern, copy it
        elif os.path.isfile(src_item) and (
            item.startswith('HMDB') or
            (item.startswith('SEL_') and item.endswith('.sdf'))
        ):
            shutil.copy2(src_item, dst_item)

# Define source and destination directories
#src_folder  = '/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/12_HMDB/HMDB_DFT_ICOLOS_gaus_10'
#dst_folder  = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/12_HMDB/HMDB_DFT_ICOLOS_gaus_10_3"
#copy_specific_files(src_folder, dst_folder)
    

def convert_to_standardized_smiles(smiles):
    """
    Standardize smiles for Molformer
    param smiles: A SMILES string.
    return: A SMILES string.
    """
    mol = MolFromSmiles(smiles, sanitize=False)
    mol = ChargeParent(mol)
    smol = Standardizer().standardize(mol)
    smi = MolToSmiles(smol, isomericSmiles=True)
    if '[H]' in smi:
        return convert_to_standardized_smiles(smi)
    else:
        return smi
    
    
