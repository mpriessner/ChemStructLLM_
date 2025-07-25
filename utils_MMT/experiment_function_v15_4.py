# Experimental Function

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Union, Tuple



########################################################################
################################ 5.1 ###################################
########################################################################
def load_data_smi_list(file_path, sample_size=None):
    df = pd.read_csv(file_path)
    smiles_list = df['SMILES'].tolist()
    if sample_size and sample_size < len(smiles_list):
        return np.random.choice(smiles_list, size=sample_size, replace=False)
    return smiles_list

def calculate_fingerprints(smiles_list):
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Calculating fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp.ToBitString())
    return np.array([list(map(int, fp)) for fp in fingerprints])

def perform_dimensionality_reduction(fingerprints):
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(fingerprints)

    print("Performing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(fingerprints)

    print("Performing UMAP...")
    umap_reducer = umap.UMAP(random_state=42)
    umap_result = umap_reducer.fit_transform(fingerprints)

    return tsne_result, pca_result, umap_result


def plot_tsne_umap_pca(results, labels, title, methods, save_folder):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=16)

    for ax, result, method in zip(axes, results, methods):
        zinc_mask = labels == 'ZINC'
        pubchem_mask = labels == 'PubChem'
       
        ax.scatter(result[zinc_mask, 0], result[zinc_mask, 1], c='blue', label='ZINC', alpha=0.1)
        ax.scatter(result[pubchem_mask, 0], result[pubchem_mask, 1], c='red', label='PubChem', alpha=1.0)
       
        ax.legend()
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
   
    # Save the figure in the specified folder
    save_path = os.path.join(save_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()


    

########################################################################
################################ 5.2 ###################################
########################################################################
import pandas as pd
import ast  # For safely evaluating strings containing Python literals
from sklearn.neighbors import NearestNeighbors
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import math
import pandas as pd
import torch
import ast
from tqdm import tqdm
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from tqdm import tqdm
from sklearn.neighbors import RadiusNeighborsRegressor
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import json

import utils_MMT.validate_generate_MMT_v15_4 as vgmmt #
import utils_MMT.helper_functions_pl_v15_4 as hf


def load_json_dics():
    with open('./itos.json', 'r') as f:
        itos = json.load(f)
    with open('./stoi.json', 'r') as f:
        stoi = json.load(f)
    with open('./stoi_MF.json', 'r') as f:
        stoi_MF = json.load(f)
    with open('./itos_MF.json', 'r') as f:
        itos_MF = json.load(f)    
    return itos, stoi, stoi_MF, itos_MF
    
itos, stoi, stoi_MF, itos_MF = load_json_dics()
rand_num = str(random.randint(1, 10000000))


def vectorize_db(config, stoi, stoi_MF, type_data, mode):
    path = config.vector_db
    model_MMT, dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF, single=False, mode=mode)
    len(dataloader)
    #import IPython; IPython.embed();

    """
    mode = "val"
    CLIP_model, val_dataloader = rbgvm.load_data_and_CLIP_model(config, 
                                                                mode, 
                                                                stoi, 
                                                                stoi_MF, 
                                                                itos, 
                                                                itos_MF)

    model_MMT = CLIP_model.CLIP_model.MT_model"""
    smi_list =[]
    fp_list = []
    for i, data_dict in tqdm(enumerate(dataloader)):

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT, data_dict, config)
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI.T[1:], itos)
        smi_list.extend(trg_conv_SMI)
        fp_list.extend(fingerprint.to("cpu"))

    # Assuming fp_list contains tensors, convert them to a list of lists (or strings)
    # Here, I'm converting each tensor to a list of its values
    fp_list_converted = [fp.tolist() for fp in fp_list]

    # Create a DataFrame
    df = pd.DataFrame({
        'SMILES': smi_list,
        'Fingerprints': fp_list_converted})

    # Save to CSV
    df.to_csv(path, index=False)
    if type_data == "db":
        config.vector_db = path
    elif type_data == "unknown":
        config.secret_csv_SMI_vectors = path
    return config

# Load Vector database
def load_vector_db(path, number):

    # Load the DataFrame from CSV
    print("Load Vector Database")
    df_fp_list = pd.read_csv(path)
    print("Vector Database finished loading")

    df_fp_list =  df_fp_list[:number]
    smi_list = list(df_fp_list['SMILES'])

    tqdm.pandas()
    # Convert the string representation of lists back to actual lists
    df_fp_list['Fingerprints'] = df_fp_list['Fingerprints'].progress_apply(ast.literal_eval)

    # Convert lists to PyTorch tensors
    df_fp_list['Fingerprints'] = df_fp_list['Fingerprints'].progress_apply(lambda x: torch.tensor(x, dtype=torch.float32))

    
    fp_tensor_list = []
    fp_list = []
    count = len(df_fp_list)
    for idx in tqdm(range(count)):
        fp = df_fp_list['Fingerprints'][idx]
        fp_tensor_list.append(fp)
        fp_list.append(list(fp))

    fp_tensor = torch.stack(fp_tensor_list)
    # Convert the tensor to a NumPy array if needed
    database_vectors = fp_tensor.numpy()
    return fp_tensor, database_vectors, smi_list

def vectorize_db_pkl(config, stoi, stoi_MF, type_data, mode):
    path = config.vector_db
    model_MMT, dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF, single=False, mode=mode)
    len(dataloader)
    """
    mode = "val"
    CLIP_model, val_dataloader = rbgvm.load_data_and_CLIP_model(config, 
                                                                mode, 
                                                                stoi, 
                                                                stoi_MF, 
                                                                itos, 
                                                                itos_MF)
    model_MMT = CLIP_model.CLIP_model.MT_model
    """
    smi_list =[]
    fp_list = []
    for i, data_dict in tqdm(enumerate(dataloader)):

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, _= vgmmt.run_model(model_MMT, data_dict, config)
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI.T[1:], itos)
        smi_list.extend(trg_conv_SMI)
        fp_list.extend(fingerprint.to("cpu"))

    # Create a DataFrame
    df = pd.DataFrame({
        'SMILES': smi_list,
        'Fingerprints': fp_list})
    #data_list = [smi_list, fp_list]
    
    with open(path, 'wb') as f:
        pickle.dump(df, f)

    if type_data == "db":
        config.vector_db = path
    elif type_data == "unknown":
        config.secret_pkl_SMI_vectors = path
    return config


def load_vector_db_pkl(pickle_file, number):
    start_time_pickle = time.time()
    
    with open(pickle_file, 'rb') as f:
        df = pickle.load(f)
    end_time_pickle = time.time()
    pickle_load_time = end_time_pickle - start_time_pickle

    df_sample = df.sample(n=number, random_state=42)

    print(f"Time taken to load Pickle: {pickle_load_time:.4f} seconds")
    print("Data processed successfully")

    df_sample = df_sample.to_dict(orient='list')
    fingerprint_batches = df_sample['Fingerprints']
    fp_tensor = torch.stack(fingerprint_batches)
    smi_list = list(df_sample['SMILES'])

    return fp_tensor, smi_list

def plot_smis_in_rows(similar_compounds_smiles):
    # Convert SMILES to RDKit molecule objects
    molecules = [Chem.MolFromSmiles(smile) for smile in similar_compounds_smiles]

    # Determine the number of rows needed for 5 molecules per row
    num_molecules = len(molecules)
    num_rows = math.ceil(num_molecules / 5)

    # Set up the figure for plotting
    plt.figure(figsize=(15, num_rows * 3))  # Adjust the figure size as needed

    # Plot the molecules in a grid layout
    for i, mol in enumerate(molecules):
        plt.subplot(num_rows, 5, i+1)  # 5 molecules per row
        img = Draw.MolToImage(mol)
        plt.imshow(img)
        plt.axis('off')  # No axes for individual plots

    plt.tight_layout()
    plt.show()

def get_fingerprint_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return None

from rdkit import DataStructs

def numpy_to_bitvect(fp_numpy):
    bitvect = DataStructs.ExplicitBitVect(len(fp_numpy))
    for i, bit in enumerate(fp_numpy):
        if bit:
            bitvect.SetBit(i)
    return bitvect

# Function to find KNN
def find_knn(database_vectors, smi_list, new_compound_vector, neighbors, extra_num):
    nn_model = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree')
    nn_model.fit(np.array(list(database_vectors)))

    new_compound_vector = np.array(new_compound_vector).reshape(1, -1)
    distances, indices = nn_model.kneighbors(new_compound_vector)

    similar_compounds_smiles = [smi_list[i] for i in indices[0]]
    similar_compounds_vectors = [database_vectors[i] for i in indices[0]]

    all_other_indices = [i for i in range(len(smi_list)) if i not in indices[0]]
    random.shuffle(all_other_indices)
    other_smiles_indices = all_other_indices[:extra_num]
    other_smiles = [smi_list[i] for i in other_smiles_indices]
    other_compounds_vectors = [database_vectors[i] for i in other_smiles_indices]

    return similar_compounds_smiles, similar_compounds_vectors, other_smiles, other_compounds_vectors, distances[0]



def find_cos_sim_incremental(database_vectors, smi_list, new_compound_vector, k, batch_size=10000):
    # Ensure inputs are NumPy arrays
    database_vectors = np.asarray(database_vectors)
    new_compound_vector = np.asarray(new_compound_vector).reshape(1, -1)  # Ensure it's 2D

    n_samples = len(database_vectors)
    all_cosine_similarities = np.zeros(n_samples)

    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch = database_vectors[i:i+batch_size]
        batch_similarities = 1 - np.array([cosine(new_compound_vector, vec) for vec in batch])
        all_cosine_similarities[i:i+batch_size] = batch_similarities

    # Get indices of top k similar compounds
    top_k_indices = np.argsort(all_cosine_similarities)[-k:][::-1]

    # Get the corresponding SMILES and vectors
    similar_compounds_smiles = [smi_list[i] for i in top_k_indices]
    similar_compounds_vectors = database_vectors[top_k_indices]
    similar_cosine_similarities = all_cosine_similarities[top_k_indices]

    return similar_compounds_smiles, similar_compounds_vectors, similar_cosine_similarities, all_cosine_similarities

# Similarly, update find_knn_incremental and find_dot_product_incremental
def find_knn_incremental(database_vectors, smi_list, new_compound_vector, k, extra_num, batch_size=10000):
    database_vectors = np.asarray(database_vectors)
    new_compound_vector = np.asarray(new_compound_vector).reshape(1, -1)

    n_samples = len(database_vectors)
    all_distances = np.zeros(n_samples)

    for i in range(0, n_samples, batch_size):
        batch = database_vectors[i:i+batch_size]
        batch_distances = np.linalg.norm(batch - new_compound_vector, axis=1)
        all_distances[i:i+batch_size] = batch_distances

    top_k_indices = np.argsort(all_distances)[:k]

    similar_compounds_smiles = [smi_list[i] for i in top_k_indices]
    similar_compounds_vectors = database_vectors[top_k_indices]
    euc_distances = all_distances[top_k_indices]

    return similar_compounds_smiles, similar_compounds_vectors, None, None, euc_distances

def find_dot_product_incremental(database_vectors, smi_list, new_compound_vector, k, batch_size=10000):
    database_vectors = np.asarray(database_vectors)
    new_compound_vector = np.asarray(new_compound_vector).reshape(1, -1)

    n_samples = len(database_vectors)
    all_dot_products = np.zeros(n_samples)

    for i in range(0, n_samples, batch_size):
        batch = database_vectors[i:i+batch_size]
        batch_dot_products = np.dot(batch, new_compound_vector.T).flatten()
        all_dot_products[i:i+batch_size] = batch_dot_products

    top_k_indices = np.argsort(all_dot_products)[-k:][::-1]

    similar_compounds_smiles = [smi_list[i] for i in top_k_indices]
    similar_compounds_vectors = database_vectors[top_k_indices]
    similar_dot_similarities = all_dot_products[top_k_indices]

    return similar_compounds_smiles, similar_compounds_vectors, similar_dot_similarities
    
    
# Function to get Morgan fingerprint vector
def get_Mfingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    return None


# Function to calculate Euclidean distance
def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def find_optimal_radius(database_vectors, new_compound_vector, target_neighbors=30, tolerance=5, max_iterations=100):
    radius = 1.0  # Initial radius guess
    step_size = 0.5  # Initial step size for adjusting the radius

    for iteration in range(max_iterations):
        nn_model = RadiusNeighborsRegressor(radius=radius)
        nn_model.fit(database_vectors, database_vectors)  # Y is not used in this case

        indices = nn_model.radius_neighbors(new_compound_vector.reshape(1, -1), return_distance=False)[0]
        num_neighbors = len(indices)

        if abs(num_neighbors - target_neighbors) <= tolerance:
            # If the number of neighbors is within the acceptable range
            return radius

        # Adjust the radius based on the number of neighbors
        if num_neighbors < target_neighbors:
            radius += step_size  # Increase the radius
        else:
            radius -= step_size  # Decrease the radius

        # Gradually reduce the step size for finer adjustments
        step_size *= 0.9

    return radius

def find_neighbors_within_threshold(database_vectors, smi_list, new_compound_vector, radius, extra_num):
    
    # Using RadiusNeighbors from scikit-learn
    nn_model = NearestNeighbors(radius=radius, algorithm='ball_tree')
    nn_model.fit(database_vectors)

    # Find the neighbors within the specified radius
    new_compound_vector = new_compound_vector.reshape(1, -1)
    indices = nn_model.radius_neighbors(new_compound_vector, return_distance=False)[0]

    # Retrieve the corresponding SMILES notations
    similar_compounds_smiles = [smi_list[i] for i in indices]
    similar_compounds_vectors = [database_vectors[i] for i in indices]

    # Get other SMILES strings that are not in the indices list
    other_smiles = [smi for i, smi in enumerate(smi_list) if i not in indices]
    other_compounds_vectors  = [vec for i, vec in enumerate(database_vectors) if i not in indices]
    other_smiles_ = other_smiles[:extra_num]
    other_compounds_vectors_ = other_compounds_vectors[:extra_num]

    return similar_compounds_smiles, similar_compounds_vectors, other_smiles_, other_compounds_vectors_


# Plotting function
def plot_smiles(smiles_list, per_row=5):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    rows = len(smiles_list) // per_row + int(len(smiles_list) % per_row != 0)
    
    fig, axes = plt.subplots(rows, per_row, figsize=(20, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for ax, mol in zip(axes, mols):
        if mol:
            img = Draw.MolToImage(mol)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_average_tanimoto(smiles_list, target_smiles):
    # Convert target SMILES to molecule and then to fingerprint
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2)  # Using bit vector fingerprints

    similarities = []

    # Calculate Tanimoto similarity for each SMILES in the list
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Ensure the molecule could be parsed from SMILES
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            # Calculate the Tanimoto similarity and append to the list
            sim = TanimotoSimilarity(target_fp, fp)
            similarities.append(sim)
        else:
            # Handle the case where a molecule could not be parsed
            similarities.append(0)

    # Calculate the average similarity
    average_similarity = np.mean(similarities)

    return average_similarity, similarities


def plot_hist_of_results_greedy_new(results_dict):
    # Extract the Tanimoto similarities
    tani_list = results_dict.get('tanimoto_sim', [])

    fig, ax = plt.subplots()

    # Create histogram
    ax.hist(tani_list, bins=20, edgecolor='black')

    # Add title and labels
    plt.title(f'Histogram of Greedy Sampled Tanimoto Similarity: {len(tani_list)} Molecules')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

    avg_tani = sum(tani_list) / len(tani_list) if tani_list else 0
    return avg_tani


def load_pickle_data(file_path, sample_size=None):
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    if isinstance(df, pd.DataFrame):
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        vectors = np.stack(df['Fingerprints'].apply(lambda x: x.numpy()).values)
        smiles = df['SMILES'].tolist()
    else:
        vectors = np.array([v.numpy() for v in df.values()])
        smiles = list(df.keys())
    
    return vectors, smiles



def plot_tsne_umap_pca_train_test(results, labels, title, methods, save_folder):
    plt.rcParams.update({'font.size': 22})  # Set base font size
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(title, fontsize=22)

    for ax, result, method in zip(axes, results, methods):
        for label, color, alpha in zip(set(labels), ['#A1C8F3', '#FFB381'], [0.5, 0.5]):
            mask = np.array(labels) == label
            ax.scatter(result[mask, 0], result[mask, 1], c=color, label=label, alpha=alpha)
       
        ax.legend(fontsize=22)  # Set legend font size
        ax.set_title(method, fontsize=22, pad=20)  # Set subplot title font size and add padding
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    #plt.savefig(f"{title.replace(' ', '_')}.png")
    # save_folder = "/projects/cc/se_users/knlr326/1_NMR_project/1_NMR_data_AZ/___FIGURES_PAPERS/Figures_Paper_2"
    save_path = os.path.join(save_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)

    plt.close()




def plot_tsne_umap_pca_train_test_folder(results, labels, title, methods, save_folder):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=16)
    colors = ['blue', 'red', 'green', 'orange']
    alphas = [0.1, 1.0, 1.0, 1.0]
    
    for ax, result, method in zip(axes, results, methods):
        for i, label in enumerate(set(labels)):
            mask = np.array(labels) == label
            ax.scatter(result[mask, 0], result[mask, 1], c=colors[i], label=label, alpha=alphas[i])
        
        ax.legend()
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the figure in the specified folder
    save_path = os.path.join(save_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Plot saved to: {save_path}")

    
    
########################################################################
################################ 5.4 ###################################
########################################################################

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap   
import os
import pickle
import numpy as np
import pandas as pd
import random
import string

def load_data_results(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data["results_dict_bl_ZINC"]


def process_pkl_files_baseline(folder_path, ranking_method):
    pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.endswith('.pkl')]
    #print(f"Processing {len(pkl_files)} files in {folder_path}")

    all_rankings = defaultdict(list)

    for file_path in pkl_files:
        file_data = load_data_results(file_path)
        try:
            ranked_molecules = rank_molecules_in_file(file_data, ranking_method)
            trg_smi = ranked_molecules[0][0]      
            for molecule in ranked_molecules:
                all_rankings[trg_smi].append(molecule)
        except:
            print(file_path)
            continue
            #import IPython; IPython.embed()
            #pass

    print(f"Processed {len(all_rankings)} unique molecules")
    return all_rankings



def analyze_molecules_by_sim_rank(all_rankings):
    count_rank_one = 0
    smiles_rank_one = []
    smiles_not_rank_one = []
    
    for rankings in all_rankings.values():
        rank_one_found = False
        for molecule in rankings:
            if molecule[2] == 1.0 and not rank_one_found:
                count_rank_one += 1
                smiles_rank_one.append(molecule[0])
                rank_one_found = True
            elif molecule[2] != 1.0:
                smiles_not_rank_one.append(molecule[0])
            break
    return count_rank_one, smiles_rank_one, smiles_not_rank_one

def perform_dimensionality_reduction(vectors):
    tsne = TSNE(n_components=2, random_state=42)
    pca = PCA(n_components=2, random_state=42)
    umap_reducer = umap.UMAP(n_components=2, random_state=42)

    tsne_result = tsne.fit_transform(vectors)
    pca_result = pca.fit_transform(vectors)
    umap_result = umap_reducer.fit_transform(vectors)

    return tsne_result, pca_result, umap_result

def plot_results(results, labels, title, methods, output_folder):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=16)

    colors = ['red', 'blue', 'green']
    alphas = [0.5, 0.2, 0.5]

    for ax, result, method in zip(axes, results, methods):
        for label, color, alpha in zip(set(labels), colors, alphas):
            mask = np.array(labels) == label
            ax.scatter(result[mask, 0], result[mask, 1], c=color, label=label, alpha=alpha)
        
        ax.legend()
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_file = os.path.join(output_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot to {output_file}")


def generate_random_id(length=10):
    """Generate a random alphanumeric ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def save_examples_as_csv(smiles_list, output_path, prefix):
    """Save SMILES as CSV with random sample IDs."""
    data = {
        'sample_id': [f"{prefix}_{generate_random_id()}" for _ in smiles_list],
        'SMILES': smiles_list
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(smiles_list)} examples to {output_path}")
    
def process_dataset(data_config, train_vectors, train_smiles, output_folder, ranking_method):
    weight_range = data_config['weight_range']
    pkl_folder = data_config['pkl_folder']
    file_path = data_config['file_path']

    all_rankings = process_pkl_files_baseline(pkl_folder, ranking_method)
    all_rankings, removed_smiles = deduplicate_smiles_from_ranking(all_rankings)
    all_rankings, filtered_out_rankings = filter_rankings_by_molecular_formula(all_rankings)

    accuracies = calculate_top_k_accuracy(all_rankings)
    count_rank_one, smiles_rank_one, smiles_not_rank_one = analyze_molecules_by_sim_rank(all_rankings)

    vectors, smiles, df = load_pickle_data_df(file_path)

    df_filtered_positive = df[df["SMILES"].isin(smiles_rank_one)]
    df_filtered_negative = df[df["SMILES"].isin(smiles_not_rank_one)]
    test_vectors_sample_positive = np.stack(df_filtered_positive['Fingerprints'].apply(lambda x: x.numpy()).values)
    test_vectors_sample_negative = np.stack(df_filtered_negative['Fingerprints'].apply(lambda x: x.numpy()).values)

    combined_vectors = np.vstack((train_vectors, test_vectors_sample_positive, test_vectors_sample_negative))
    labels = ['ZINC'] * len(train_vectors) + [f'{weight_range}_pos'] * len(test_vectors_sample_positive) + [f'{weight_range}_neg'] * len(test_vectors_sample_negative)

    print(f"Performing dimensionality reduction for {weight_range}...")
    tsne_result, pca_result, umap_result = perform_dimensionality_reduction(combined_vectors)

    print(f"Plotting results for {weight_range}...")
    plot_results([tsne_result, pca_result, umap_result], labels, 
                 f"Train vs Test Vectors ({weight_range})", 
                 ['t-SNE', 'PCA', 'UMAP'],
                 output_folder)

def process_dataset_(data_config, train_vectors, train_smiles, output_folder, ranking_method):
    weight_range = data_config['weight_range']
    pkl_folder = data_config['pkl_folder']
    file_path = data_config['file_path']

    all_rankings = process_pkl_files_baseline(pkl_folder, ranking_method)
    all_rankings, removed_smiles = deduplicate_smiles_from_ranking(all_rankings)
    all_rankings, filtered_out_rankings = filter_rankings_by_molecular_formula(all_rankings)

    accuracies = calculate_top_k_accuracy(all_rankings)
    count_rank_one, smiles_rank_one, smiles_not_rank_one = analyze_molecules_by_sim_rank(all_rankings)

    # Load the pickle data
    vectors, smiles, df = load_pickle_data_df(file_path)
    
    # Filter vectors by finding indices of SMILES in the original lists
    test_vectors_sample_positive, test_vectors_sample_negative = filter_vectors_by_smiles(
        vectors, smiles, smiles_rank_one, smiles_not_rank_one
    )

    combined_vectors = np.vstack((train_vectors, test_vectors_sample_positive, test_vectors_sample_negative))
    labels = ['ZINC'] * len(train_vectors) + [f'{weight_range}_pos'] * len(test_vectors_sample_positive) + [f'{weight_range}_neg'] * len(test_vectors_sample_negative)

    print(f"Performing dimensionality reduction for {weight_range}...")
    tsne_result, pca_result, umap_result = perform_dimensionality_reduction(combined_vectors)

    print(f"Plotting results for {weight_range}...")
    plot_results([tsne_result, pca_result, umap_result], labels, 
                 f"Train vs Test Vectors_ ({weight_range})", 
                 ['t-SNE', 'PCA', 'UMAP'],
                 output_folder)


def filter_vectors_by_smiles(vectors, smiles_list, smiles_positive, smiles_negative):
    """
    Filter vectors by finding the indices of SMILES in the original lists.
    
    Args:
        vectors: numpy array of vectors
        smiles_list: list of SMILES strings corresponding to the vectors
        smiles_positive: list of SMILES that should be in the positive set
        smiles_negative: list of SMILES that should be in the negative set
    
    Returns:
        tuple: (positive_vectors, negative_vectors)
    """
    # Convert to sets for faster lookup
    smiles_positive_set = set(smiles_positive)
    smiles_negative_set = set(smiles_negative)
    
    # Find indices of positive and negative SMILES
    positive_indices = []
    negative_indices = []
    
    for i, smile in enumerate(smiles_list):
        if smile in smiles_positive_set:
            positive_indices.append(i)
        elif smile in smiles_negative_set:
            negative_indices.append(i)
    
    # Extract vectors at these indices
    test_vectors_sample_positive = vectors[positive_indices] if positive_indices else np.empty((0, vectors.shape[1]))
    test_vectors_sample_negative = vectors[negative_indices] if negative_indices else np.empty((0, vectors.shape[1]))
    
    print(f"Found {len(positive_indices)} positive samples and {len(negative_indices)} negative samples")
    print(f"Positive vector shape: {test_vectors_sample_positive.shape}")
    print(f"Negative vector shape: {test_vectors_sample_negative.shape}")
    
    return test_vectors_sample_positive, test_vectors_sample_negative


# Alternative version with more detailed logging
def filter_vectors_by_smiles_verbose(vectors, smiles_list, smiles_positive, smiles_negative):
    """
    Filter vectors by finding the indices of SMILES in the original lists with verbose logging.
    """
    # Convert to sets for faster lookup
    smiles_positive_set = set(smiles_positive)
    smiles_negative_set = set(smiles_negative)
    
    print(f"Total vectors: {len(vectors)}")
    print(f"Total SMILES: {len(smiles_list)}")
    print(f"Target positive SMILES: {len(smiles_positive_set)}")
    print(f"Target negative SMILES: {len(smiles_negative_set)}")
    
    # Find indices of positive and negative SMILES
    positive_indices = []
    negative_indices = []
    found_positive = set()
    found_negative = set()
    
    for i, smile in enumerate(smiles_list):
        if smile in smiles_positive_set:
            positive_indices.append(i)
            found_positive.add(smile)
        elif smile in smiles_negative_set:
            negative_indices.append(i)
            found_negative.add(smile)
    
    # Report matching statistics
    print(f"Found {len(found_positive)}/{len(smiles_positive_set)} positive SMILES")
    print(f"Found {len(found_negative)}/{len(smiles_negative_set)} negative SMILES")
    
    missing_positive = smiles_positive_set - found_positive
    missing_negative = smiles_negative_set - found_negative
    
    if missing_positive:
        print(f"Missing positive SMILES: {len(missing_positive)} (showing first 5): {list(missing_positive)[:5]}")
    if missing_negative:
        print(f"Missing negative SMILES: {len(missing_negative)} (showing first 5): {list(missing_negative)[:5]}")
    
    # Extract vectors at these indices
    test_vectors_sample_positive = vectors[positive_indices] if positive_indices else np.empty((0, vectors.shape[1]))
    test_vectors_sample_negative = vectors[negative_indices] if negative_indices else np.empty((0, vectors.shape[1]))
    
    print(f"Positive vector shape: {test_vectors_sample_positive.shape}")
    print(f"Negative vector shape: {test_vectors_sample_negative.shape}")
    
    return test_vectors_sample_positive, test_vectors_sample_negative
    

def rank_molecules_in_file(file_data, ranking_method):
    molecule_data = []
    for trg_smi, value_list in file_data.items():
        try:
            for sublist in value_list[0]:
                gen_smi = sublist[0]
                tanimoto = sublist[4]
                errors = sublist[5]
                if errors == 9:
                    errors = [9,9] ### Necessary if it doesn't manage to calculate HSQC or COSY to calculate the errors
                                    ## Basically put it last then
                molecule_data.append((trg_smi, gen_smi, errors[0], errors[1], tanimoto))
        except:
            pass
            #import IPython; IPython.embed()
    if not molecule_data:
        return []

    molecule_array = np.array(molecule_data, dtype=[('trg_smi', 'U100'), ('gen_smi', 'U100'), 
                                                    ('error1', float), ('error2', float), 
                                                    ('tanimoto', float)])
    
    if ranking_method == 'HSQC & COSY':
        rank1 = molecule_array.argsort(order='error1')
        rank2 = molecule_array.argsort(order='error2')
        average_ranks = (np.arange(len(rank1))[np.argsort(rank1)] + 
                         np.arange(len(rank2))[np.argsort(rank2)]) / 2
        sorted_indices = average_ranks.argsort()
        
    elif ranking_method == 'HSQC':
        sorted_indices = molecule_array.argsort(order='error1')
    elif ranking_method == 'COSY':
        sorted_indices = molecule_array.argsort(order='error2')
    else:
        raise ValueError("Invalid ranking method. Choose 'HSQC & COSY', 'HSQC', or 'COSY'.")

    sorted_molecules = [(molecule_array['trg_smi'][i], 
                         molecule_array['gen_smi'][i],
                         molecule_array["tanimoto"][i],
                         new_rank,  # Use index as rank
                         molecule_array['error1'][i], 
                         molecule_array['error2'][i]) 
                        for new_rank, i  in enumerate(sorted_indices)]
    return sorted_molecules

    
########################################################################
################################ 5.5 ###################################
########################################################################


def generate_random_id(length=10):
    """Generate a random alphanumeric ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def save_examples_as_csv(smiles_list, output_path, prefix):
    """Save SMILES as CSV with random sample IDs."""
    data = {
        'sample_id': [f"{prefix}_{generate_random_id()}" for _ in smiles_list],
        'SMILES': smiles_list
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(smiles_list)} examples to {output_path}")

def process_dataset_and_save_csv(data_config, output_folder, ranking_method):
    weight_range = data_config['weight_range']
    pkl_folder = data_config['pkl_folder']
    #file_path = data_config['file_path']

    all_rankings = process_pkl_files_baseline(pkl_folder, ranking_method)
    all_rankings, removed_smiles = deduplicate_smiles_from_ranking(all_rankings)
    all_rankings, filtered_out_rankings = filter_rankings_by_molecular_formula(all_rankings)

    count_rank_one, smiles_rank_one, smiles_not_rank_one = analyze_molecules_by_sim_rank(all_rankings)

    # Save positive examples
    pos_csv_path = os.path.join(output_folder, f"{weight_range}_positive.csv")
    save_examples_as_csv(smiles_rank_one, pos_csv_path, f"{weight_range}_pos")

    # Save negative examples
    neg_csv_path = os.path.join(output_folder, f"{weight_range}_negative.csv")
    save_examples_as_csv(smiles_not_rank_one, neg_csv_path, f"{weight_range}_neg")

    return pos_csv_path, neg_csv_path


def deduplicate_smiles_from_ranking(all_rankings):
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else smiles
        #return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True) if mol else smiles

    deduplicated_rankings = {}
    removed_smiles = {}

    for key, rankings in all_rankings.items():
        seen_canonical = set()
        deduplicated_rankings[key] = []
        removed_smiles[key] = []

        for ranking in rankings:
            canonical_smiles = canonicalize(ranking[1])
            if canonical_smiles not in seen_canonical:
                seen_canonical.add(canonical_smiles)
                deduplicated_rankings[key].append(ranking)
            else:
                removed_smiles[key].append(ranking[1])

    return deduplicated_rankings, removed_smiles


def filter_rankings_by_molecular_formula(all_rankings):
    filtered_rankings = defaultdict(list)
    filtered_out_rankings = defaultdict(list)

    for key, rankings in all_rankings.items():
        filtered_rankings_for_key = []
        filtered_out_rankings_for_key = []
        
        for ranking in rankings:
            if len(ranking) >= 2:
                smiles1 = ranking[0]
                smiles2 = ranking[1]
                formula1 = get_molecular_formula(smiles1)
                formula2 = get_molecular_formula(smiles2)
                
                if formula1 is not None and formula2 is not None and formula1 == formula2:
                    filtered_rankings_for_key.append(ranking)
                else:
                    filtered_out_rankings_for_key.append(ranking)
        
        if filtered_rankings_for_key:
            filtered_rankings[key] = filtered_rankings_for_key
        
        if filtered_out_rankings_for_key:
            filtered_out_rankings[key] = filtered_out_rankings_for_key

    return filtered_rankings, filtered_out_rankings


def get_molecular_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.rdMolDescriptors.CalcMolFormula(mol)


def calculate_top_k_accuracy_BL(all_rankings, k_range=[1, 3, 5, 10, 20]):
    accuracies = []
    total_molecules = len(all_rankings)
    for k in k_range:
        correct_count = sum(
            any(molecule[2] == 1.0 for molecule in rankings[:k])
            for rankings in all_rankings.values()
        )
        accuracy = correct_count / total_molecules
        accuracies.append(accuracy)
    return accuracies

def calculate_top_k_accuracy(all_rankings, k_range=[1, 3, 5, 10, 20]):
    accuracies = []
    total_molecules = len(all_rankings)
    for k in k_range:
        correct_count = sum(
            any(molecule[2] == 1 for molecule in rankings[:k])
            for rankings in all_rankings.values()
        )
        accuracy = correct_count / total_molecules
        accuracies.append(accuracy)
    correct_count = sum(
        any(molecule[2] == 1 for molecule in rankings[:])
        for rankings in all_rankings.values()
    )
    accuracy = correct_count / total_molecules
    accuracies.append(accuracy)    
    return accuracies



def rank_molecules(file_data, ranking_method):
    all_rankings = {}
    for trg_smi, value_list in file_data.items():
        molecule_data = []
        for sublist in value_list[0]:
            try:
                gen_smi = sublist[0]
                tanimoto = sublist[4]
                errors = sublist[5]
                molecule_data.append((gen_smi, errors[0], errors[1], tanimoto))
            except:
                print(f"Error processing sublist for {trg_smi}: {sublist}")
                continue

        if not molecule_data:
            continue

        molecule_array = np.array(molecule_data, dtype=[('gen_smi', 'U100'), 
                                                        ('error1', float), ('error2', float), 
                                                        ('tanimoto', float)])
        
        if ranking_method == 'HSQC & COSY':
            rank1 = molecule_array.argsort(order='error1')
            rank2 = molecule_array.argsort(order='error2')
            average_ranks = (np.arange(len(rank1))[np.argsort(rank1)] + 
                             np.arange(len(rank2))[np.argsort(rank2)]) / 2
            sorted_indices = average_ranks.argsort()
        elif ranking_method == 'HSQC':
            sorted_indices = molecule_array.argsort(order='error1')
        elif ranking_method == 'COSY':
            sorted_indices = molecule_array.argsort(order='error2')
        else:
            raise ValueError("Invalid ranking method. Choose 'HSQC & COSY', 'HSQC', or 'COSY'.")

        sorted_molecules = [(trg_smi,
                             molecule_array['gen_smi'][i],
                             molecule_array["tanimoto"][i],
                             new_rank,  # Use index as rank
                             molecule_array['error1'][i], 
                             molecule_array['error2'][i]) 
                            for  new_rank, i in enumerate(sorted_indices)]
        
        all_rankings[trg_smi] = sorted_molecules
    
    return all_rankings


def process_pkl_files_BL(file_paths_dict, ranking_method):
    all_rankings = {}
    for data_type, file_path in file_paths_dict.items():
        file_data = load_data_results(file_path)
        all_rankings[data_type] = rank_molecules(file_data, ranking_method)
        
    return all_rankings



def count_molecules_with_sim_rank_one(all_rankings):
    count = sum(
        any(molecule[2] == 1.0 for molecule in rankings)
        for rankings in all_rankings.values()
    )
    return count


def load_pickle_data_df(file_path, sample_size=None):
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    vectors = np.stack(df['Fingerprints'].apply(lambda x: x.numpy()).values)
    smiles = df['SMILES'].tolist()
    return vectors, smiles, df



def rename_column_in_csv(file_path, old_column_name, new_column_name):
    """
    Rename a column in a CSV file and save it back to the same file.
    
    :param file_path: Path to the CSV file
    :param old_column_name: Current name of the column
    :param new_column_name: New name for the column
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Rename the column
    df = df.rename(columns={old_column_name: new_column_name})
    
    # Save the DataFrame back to the same CSV file
    df.to_csv(file_path, index=False)
    
    print(f"Processed: {file_path}")
    
    
    
    
def process_and_save_histogram(experiment_results, save_dir, title_label):
    def calculate_average(tuple_list):
        return np.mean([t[1] for t in tuple_list])
    
    averages = []
    
    # Loop over all SMILES keys in the dictionary
    for smiles, data in experiment_results.items():
        # Access the list of tuples from the 'similar_compounds' key
        similar_compounds = data.get('similar_compounds', [])
        
        # Calculate the average for this SMILES and add to the list
        if similar_compounds:
            avg = calculate_average(similar_compounds)
            averages.append(avg)
    
    # Calculate the overall average
    overall_average = np.mean(averages)
    
    # Create histogram
    plt.figure(figsize=(12, 7))
    n, bins, patches = plt.hist(averages, bins=20, edgecolor='black')
    
    # Add vertical line for average
    plt.axvline(overall_average, color='r', linestyle='dashed', linewidth=2)
    
    # Add text annotation for average
    plt.text(overall_average, plt.ylim()[1], f'Average: {overall_average:.2f}', 
             horizontalalignment='center', verticalalignment='bottom')
    
    plt.title(f'Histogram of Average Similar Compounds\n{title_label}')
    plt.xlabel('Average of Similar Compounds')
    plt.ylabel('Frequency')
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Create filename
    filename = f"histogram_{title_label.replace(' ', '_')}.png"
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300)  # Increased DPI for better quality
    plt.close()  # Close the figure to free up memory
    
    print(f"Histogram saved as: {full_path}")
    
    return averages, overall_average


def process_spectrum_data(config: Any, sample_ids: List[str], data_type: str) -> None:
    spectrum_types = ['1H', '13C', 'HSQC', 'COSY']
    for spectrum in spectrum_types:
        csv_path = getattr(config, f'csv_{spectrum}_path_{data_type}')
        df_data = pd.read_csv(csv_path)
        df_data['sample-id'] = df_data['AZ_Number']
        data = select_relevant_samples(df_data, sample_ids)
        dummy_path, config = save_and_update_config(config, data_type, spectrum, data)
        print(f"Saved {spectrum} data to: {dummy_path}")
    if data_type == "ACD" or data_type == "sim":
        config.IR_data_folder = config.IR_data_folder_backup 
    elif  data_type == "exp":
        config.IR_data_folder = config.IR_data_folder_exp 