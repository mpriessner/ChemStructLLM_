from rdkit import DataStructs
from collections import Counter
import re
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import json
import statistics
import pandas as pd
from rdkit.Chem import rdMolDescriptors
import random

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  ### To avoid the warning of invalid smiles

def calculate_weight(smiles_str):
    """ used in compare_molformer input smiles string"""
    m = Chem.MolFromSmiles(smiles_str)
    return ExactMolWt(m)

def calculate_tanimoto_from_two_smiles(smi1, smi2, nbits=2024, extra_info = False):
    """This function takes two smile_stings and 
    calculates the Tanimoto similarity and returns it and prints it out"""
    
    pattern1 = Chem.MolFromSmiles(smi1)
    pattern2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(pattern2, 2, nBits=nbits)

    tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    tan_sim = round(tan_sim,4)
    if extra_info:
        print(f"Smiles 1: {smi1} \n Target Smiles: {smi2} \nTanimoto score:{tan_sim}")
    
    return tan_sim

def try_calculate_tanimoto_from_two_smiles(smi1, smi2, nbits, extra_info = False):
    """This function takes two smile_stings and 
    calculates the Tanimoto similarity and returns it and prints it out"""
    
    try:
        pattern1 = Chem.MolFromSmiles(smi1)
        pattern2 = Chem.MolFromSmiles(smi2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(pattern2, 2, nBits=nbits)

        tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        tan_sim = round(tan_sim,4)
        if extra_info:
            print(f"Smiles 1: {smi1} \n Target Smiles: {smi2} \nTanimoto score:{tan_sim}")

        return tan_sim
    except:
        return None

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])   

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol 


def get_validity_term(decoded_molecules):
    """count = 0
    total = 0
    for smi in decoded_molecules:
        valid = is_valid_smiles(smi)
        if valid != None:
            count +=1
        total += 1
    validity_score = count/total"""
    
    count = sum(1 for smi in decoded_molecules if is_valid_smiles(smi) is not None)
    total = len(decoded_molecules)
    validity_score = count / total if total > 0 else 0

    return torch.tensor(validity_score).float()

def symbol_counts(smiles):
    symbols = list(smiles)
    #symbol_regex = re.compile(r'[A-Z][a-z]*|\d|\[.*?\]|\(|\)|=|#|-|\+')
    #symbols = symbol_regex.findall(smiles)
    symbol_counts = Counter(symbols)
    return symbol_counts

'''
def count_based_reward(generated_smiles_list, target_smiles):
    """ This function counts the number of correct atoms vs incorrect atoms in 
    the generated vs real SMILES and returns the average % score """
    total_score = 0
    total_number = 0
    for generated_smi, target_smi in zip(generated_smiles_list, target_smiles):
        generated_counts = symbol_counts(generated_smi)
        target_counts = symbol_counts(target_smi)

        correct_symbols = 0
        incorrect_symbols = 0

        for symbol, count in target_counts.items():
            correct_symbols += min(count, generated_counts.get(symbol, 0))
            incorrect_symbols += abs(count - generated_counts.get(symbol, 0))

        # Calculate a reward or punishment score
        score = correct_symbols/(correct_symbols+incorrect_symbols)
        total_score += score
        total_number+=1
    avg_score = total_score/total_number
    return avg_score'''

def count_based_reward(generated_smiles_list, target_smiles):
    """ This function counts the number of correct atoms vs incorrect atoms in 
    the generated vs real SMILES and returns the average % score """
    total_score = 0

    for generated_smi, target_smi in zip(generated_smiles_list, target_smiles):
        generated_counts = symbol_counts(generated_smi)
        target_counts = symbol_counts(target_smi)

        correct_symbols = sum(min(generated_counts.get(symbol, 0), count) for symbol, count in target_counts.items())

        all_symbols = set(generated_counts.keys()).union(target_counts.keys())
        incorrect_symbols = sum(abs(generated_counts.get(symbol, 0) - target_counts.get(symbol, 0)) for symbol in all_symbols)

        # Calculate a reward or punishment score
        score = correct_symbols / (correct_symbols + incorrect_symbols) if (correct_symbols + incorrect_symbols) > 0 else 0
        total_score += score

    avg_score = total_score / len(generated_smiles_list) if generated_smiles_list else 0
    return avg_score



def output_to_smiles(outputs, itos):
    smi_list = []
    for i in range(outputs.shape[1]):  # output.shape[1] = batch_size_NMR in your case
        sub_tensor = outputs[:, i, :]
    #for output in outputs:
        _, predicted_indices = torch.max(sub_tensor, dim=-1)

        smiles = ''.join([itos[str(int(i))] for i in predicted_indices])[:-5]
        smi_list.append(smiles)
    return smi_list

def calculate_gen_mol_weights(generated_smiles_list):
    weights_list = []
    for gen_smi in generated_smiles_list:
        try:
            mol = Chem.MolFromSmiles(gen_smi)
            weight = Descriptors.ExactMolWt(mol)
            weights_list.append(weight)
        except:
            weights_list.append(0.0)
    weights_tensor = torch.tensor(weights_list)
    return weights_tensor 

# Function to calculate molecular formula from SMILES string
def calculate_molecular_formula(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            return Descriptors.MolFormula(molecule)
    except:
        return None
    
    
two_char_symbols = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 
                    'Ca', 'Ti', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 
                    'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sb', 
                    'Te', 'I', 'Xe', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 
                    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 
                    'Pb', 'Bi', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
                    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh',
                    'Hs', 'Mt', 'Ds', 'Rg', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

def tokenize_smiles(smiles, two_letter_atoms):
    """
    Tokenizes the SMILES string, considering two-letter atoms.

    Parameters:
    - smiles: The SMILES string to be tokenized.
    - two_letter_atoms: A list of two-letter atoms (e.g., ['Cl', 'Br']).

    Returns:
    - tokens: A list of tokens representing the SMILES string.
    """
    tokens = []
    i = 0
    while i < len(smiles):
        # Check for two-letter atoms first
        if i < len(smiles) - 1 and smiles[i:i+2] in two_letter_atoms:
            tokens.append(smiles[i:i+2])
            i += 2
        else:
            tokens.append(smiles[i])
            i += 1
    return tokens

def calculate_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.Descriptors.MolWt(mol)
    else:
        return None
    
# Function to remove all occurrences of 'nan' from a list
# Function to remove 'nan' (both string and actual NaN) from a list
def remove_nan_from_list(lst):
    cleaned_list = []
    for x in lst:
        if isinstance(x, str) and x == 'nan':
            continue
        elif isinstance(x, float) and math.isnan(x):
            continue
        else:
            cleaned_list.append(x)
    return cleaned_list


# Function to check if a list contains only 'nan' values
import math
def contains_only_nan(lst):
    return all(isinstance(x, float) and math.isnan(x) for x in lst)



def tensor_to_smiles(tensor, itos):
    """
    Converts a tensor of token IDs to a SMILES string or list of SMILES strings.

    :param tensor: Tensor containing token IDs.
    :param itos: Dictionary mapping token IDs to SMILES tokens.
    :return: Single SMILES string or list of SMILES strings.
    """
    if len(tensor.shape)>1:
        sequence_length, batch_size = tensor.shape
        sequences = []

        for i in range(batch_size):
            sequence = []
            for j in range(0, sequence_length): 
                token = itos[str(int(tensor[j, i].item()))]
                if token == "<EOS>":
                    break
                sequence.append(token)
            sequences.append("".join(sequence))
        
        return sequences
    else: 
        sequence = []
        for j in range(0, len(tensor)): 
            token = itos[str(int(tensor[j].item()))]        
            if token == "<EOS>":
                break
            sequence.append(token)
        smi = "".join(sequence)
        return smi


def tensor_to_smiles_and_prob(tensor, token_prob, itos):
    if len(tensor.shape) > 1:
        # Convert tensor to numpy for efficient access
        tensor_np = tensor.cpu().numpy()
        sequences = []
        token_prob_cut = []

        for i in range(tensor.shape[1]):  # Iterate over batch_size
            # Find the index of the first occurrence of <EOS> token
            eos_idx = next((idx for idx, val in enumerate(tensor_np[:, i]) if itos[str(val)] == "<EOS>"), tensor.shape[0])
            
            # Build sequence using list comprehension
            sequence = [itos[str(val)] for val in tensor_np[:eos_idx, i]]
            sequences.append("".join(sequence))

            # Slice the probability tensor
            token_prob_cut.append(token_prob[i, :eos_idx])

        return sequences, token_prob_cut

    else:
        # Handle the case for a single sequence
        tensor_np = tensor.cpu().numpy()
        eos_idx = next((idx for idx, val in enumerate(tensor_np) if itos[str(val)] == "<EOS>"), len(tensor))
        
        sequence = [itos[str(val)] for val in tensor_np[:eos_idx]]
        smi = "".join(sequence)
        token_prob_list = list(token_prob[:eos_idx])

        return smi, torch.stack(token_prob_list)
    


def combine_gen_sims(gen_smi1, gen_tensor1, confidence_list_greedy, 
                     gen_smi2, gen_tensor2, confidence_list_multinom):
    """
    Combines generated SMILES strings from two different methods (greedy and multinomial).
    Prioritizes greedy method; if it fails, uses multinomial method.

    :param gen_smi1: List of SMILES strings from the greedy method.
    :param gen_tensor1: Corresponding tensor from the greedy method.
    :param confidence_list_greedy: List of confidence scores from the greedy method.
    :param gen_smi2: List of SMILES strings from the multinomial method.
    :param gen_tensor2: Corresponding tensor from the multinomial method.
    :param confidence_list_multinom: List of confidence scores from the multinomial method.
    :return: Tuple of combined tensor, SMILES strings, and confidence scores.
    """

    combined_smi = []
    combined_tensors = []
    combined_confidence = []

    for g_smi1, g_tensor1, conf_greedy, g_smi2, g_tensor2, conf_multinom in zip(
        gen_smi1, gen_tensor1.permute(1, 0), confidence_list_greedy, 
        gen_smi2, gen_tensor2.permute(1, 0), confidence_list_multinom):

        try:
            # Attempt to create a molecule from the greedy SMILES string
            Chem.MolFromSmiles(g_smi1)
            combined_smi.append(g_smi1)
            combined_tensors.append(g_tensor1)
            combined_confidence.append(conf_greedy)
        except:
            # Fall back to multinomial method if greedy fails
            combined_smi.append(g_smi2)
            combined_tensors.append(g_tensor2)
            combined_confidence.append(conf_multinom)

    # Handling the case when no successful SMILES strings were generated
    if combined_tensors:
        combined_tensor = torch.stack(combined_tensors).permute(1, 0)
    else:
        combined_tensor = None
        combined_smi = None
        combined_confidence = None

    return combined_tensor, combined_smi, combined_confidence



def calculate_tanimoto_similarity_2(gen_conv_SMI_list, trg_conv_SMI_list):
    """
    Calculates the Tanimoto similarity between generated and target SMILES strings.

    Parameters:
    gen_conv_SMI_list (list): List of generated SMILES strings.
    trg_conv_SMI_list (list): List of target SMILES strings.

    Returns:
    tuple: Mean and standard deviation of Tanimoto similarities, failed pairs, and all Tanimoto scores.
    """
    tanimoto_scores_ = []
    tanimoto_scores_all = []
    failed_pairs = []
    gen_conv_SMI_list_, trg_conv_SMI_list_, idx_list = [],[], []
    for idx, (gen_smi, trg_smi) in enumerate(zip(gen_conv_SMI_list, trg_conv_SMI_list)):

        try:
            gen_mol = Chem.MolFromSmiles(gen_smi)
            gen_smi_canonical = Chem.MolToSmiles(gen_mol, canonical=True)
            tan_sim = calculate_tanimoto_from_two_smiles(gen_smi_canonical, trg_smi, 512)
            tanimoto_scores_.append(tan_sim)
            tanimoto_scores_all.append(tan_sim)
            gen_conv_SMI_list_.append(gen_smi_canonical)
            trg_conv_SMI_list_.append(trg_smi)
            idx_list.append(idx)
        except:
            tanimoto_scores_all.append(0)
            failed_pairs.append((gen_smi, trg_smi))


    tanimoto_scores_cleaned = [i for i in tanimoto_scores_ if i!=0]
    tanimoto_mean = statistics.mean(tanimoto_scores_cleaned) if tanimoto_scores_cleaned else 0
    tanimoto_std_dev = statistics.stdev(tanimoto_scores_cleaned) if len(tanimoto_scores_cleaned) > 1 else 0

    return tanimoto_mean, tanimoto_std_dev, failed_pairs, tanimoto_scores_, tanimoto_scores_all, gen_conv_SMI_list_, trg_conv_SMI_list_, idx_list


def tensor_to_smiles_and_prob_2(tensor, token_prob, itos):

    token_prob_cut = []
    #import IPython; IPython.embed();

    if len(tensor.shape)>1 and len(token_prob.shape)>1:
        sequence_length, batch_size = tensor.shape
        sequences = []
        for i in range(batch_size):
            sequence = []
            for j in range(0, sequence_length): 
                token = itos[str(int(tensor[j, i].item()))]
                if token == "<EOS>":
                    break
                sequence.append(token)
            sequences.append("".join(sequence))
            token_prob_cut.append(token_prob[:len(sequence),i])
        return sequences, token_prob_cut

    else: 
        sequence = []
        for j in range(0, len(tensor)): 
            token = itos[str(int(tensor[j].item()))]        
            if token == "<EOS>":
                break
            sequence.append(token)
            token_prob_cut.append(token_prob[j])
        smi = "".join(sequence)
        token_prob_list = list(token_prob[:len(sequence)])
    return smi, torch.stack(token_prob_list)        


from rdkit import Chem, DataStructs
def smile2SDF(smile, folder=None, name=None):
    """ Saves a smile to a given folder with a given name/ID in sdf format
    adding the H is crutial for the ICOLOS workflow
    ir no name is provided then it saves it under a random number"""
    if name==None:
        rand_num = random.randint(0,1000)
        name = "%04d" % (rand_num)
    if folder==None:
        folder= "/projects/cc/knlr326/4_Everything/trash"
    mol = AllChem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    # mol = Chem.MolToMolBlock(mol)
    save_path = os.path.join(folder,name+".sdf")
    writer = Chem.SDWriter(save_path)
    writer.write(mol)
    writer.close()
    return save_path


from rdkit.Chem import MolFromSmiles, MolToSmiles
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
    

def calculate_tanimoto_and_mol_weights(gen_conv_SMI, trg_conv_SMI, trg_MW):
    """
    Calculate Tanimoto similarity and molecular weights for generated and target SMILES.

    Args:
        gen_conv_SMI (list): List of generated SMILES strings.
        trg_conv_SMI (list): List of target SMILES strings for comparison.
        trg_MW (list): List of molecular weights corresponding to target SMILES.

    Returns:
        tuple: Contains the mean Tanimoto similarity, lists of selected molecular weights for
               generated and target SMILES, and a DataFrame of successful generations.
    """    
    gen_mol_weights_sel = []
    trg_mol_weights_sel = []
    succ_gen_list = []
    tanimoto_list = []
    count = 0

    ##########################################
    ###### calculate Tanimoto similarty ######
    ##########################################
    for i, (gen_smi, trg_weight) in enumerate(zip(gen_conv_SMI, trg_MW)):
        try:
            gen_mol = Chem.MolFromSmiles(gen_smi)
            #gen_smi = Chem.MolToSmiles(gen_mol, canonical=True, doRandom=False, isomericSmiles=True)
            #gen_mol = Chem.MolFromSmiles(gen_smi)            
            gen_weight = rdMolDescriptors.CalcExactMolWt(gen_mol)
            gen_mol_weights_sel.append(gen_weight)
            trg_mol_weights_sel.append(trg_weight)
            tan_sim = calculate_tanimoto_from_two_smiles(gen_smi, trg_conv_SMI[i], 512, extra_info = False)  
            tanimoto_list.append(tan_sim)
            ran_num = random.randint(0, 100000)
            sample_id = str(i)+"_"+str(ran_num)
            succ_gen_list.append([sample_id, gen_smi])
            count += 1
        except:
            # Optional: Log the exception e if needed
            pass

    # Calculate mean Tanimoto similarity, avoiding division by zero
    tanimoto_mean = sum(tanimoto_list) / count if count != 0 else 0
    # Create a DataFrame of successful SMILES generations
    df_succ_smis = pd.DataFrame(succ_gen_list, columns=['sample-id', 'SMILES'])

    return tanimoto_mean, gen_mol_weights_sel, trg_mol_weights_sel, df_succ_smis