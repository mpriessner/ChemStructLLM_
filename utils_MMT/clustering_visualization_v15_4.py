from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import Image
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Standard library imports
from IPython.display import display, HTML, Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Third-party imports
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
from io import BytesIO
import base64

def display_gen_and_trg_molecules(selected_best_SMI_list, trg_conv_SMI_list, show_number):
    """Provide two lists of smiles - first column are generated molecules - second column the target
    molecules then the tanimoto similarity"""
    selected_best_mol_list = [Chem.MolFromSmiles(smiles) for smiles in tqdm(selected_best_SMI_list)]
    trg_conv_mol_list = [Chem.MolFromSmiles(smiles) for smiles in tqdm(trg_conv_SMI_list)]

    # Assume the following lists
    mol_list1  = selected_best_mol_list 
    mol_list2  = trg_conv_mol_list
    list1 = selected_best_SMI_list

    # Loop through the molecules and create a separate plot for each pair
    for i in tqdm(range(0, show_number)):
        try:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns per plot
            # Draw molecules from list 1
            img1 = Draw.MolToImage(mol_list1[i])
            axs[0].imshow(img1)
            axs[0].axis('off')

            # Draw molecules from list 2
            img2 = Draw.MolToImage(mol_list2[i])
            axs[1].imshow(img2)
            axs[1].axis('off')

            # Compute Tanimoto similarity
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_list1[i],  2, nBits=512)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_list2[i],  2, nBits=512)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            # Display Tanimoto similarity
            axs[2].text(0.5, 0.5, f'Tanimoto Similarity: {similarity:.2f}', fontsize=12, ha='center', va='center')
            axs[2].axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred for index {i}: {e}")

    
################################################################    


def generate_colored_html(smiles, probabilities):
    # Start the HTML string
    html_string = '<span style="font-family: monospace;">'

    # Iterate over each character and its associated probability
    for char, prob in zip(smiles, probabilities):
        # Calculate the RGB values for the color gradient
        red_value = int((1.0 - prob) * 255)
        green_value = int(prob * 255)
        
        # Convert RGB values to a CSS-readable format
        color = f'rgb({red_value}, {green_value}, 0)'
        
        # Append the character with its associated background color to the HTML string
        html_string += f'<span style="background-color: {color}">{char}</span>'

    # Close the HTML string
    html_string += '</span>'
    
    return html_string



# def save_molecule_to_svg(smiles, probabilities, filename="molecule.svg"):
#     mol = Chem.MolFromSmiles(smiles)
#     d = rdMolDraw2D.MolDraw2DSVG(400, 200)
#     d.drawOptions().useBWAtomPalette()
    
#     atom_colors = {i: get_color(prob.item()) for i, prob in enumerate(probabilities)}
    
#     d.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=atom_colors)
#     d.FinishDrawing()
    
#     with open(filename, "w") as f:
#         f.write(d.GetDrawingText())

def save_molecule_to_svg(smiles, probabilities):
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(400, 200)
    d.drawOptions().useBWAtomPalette()
    
    atom_colors = {i: get_color(prob.item()) for i, prob in enumerate(probabilities)}
    
    d.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=atom_colors)
    d.FinishDrawing()

    # Instead of saving to a file, we'll return the SVG as an HTML image
    svg = d.GetDrawingText().replace('\n', '')
    html = f"<img src='data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode()}'/>"
    
    return html

def get_color(prob):
    red_value = (1.0 - prob)
    green_value = prob
    return (red_value, green_value, 0)

    

def plot_2D(X, number_point_list, config, title):
    # Initialize variables
    colors = cm.get_cmap('tab20', config.n_samples+1) # Adjust the colormap size based on the number of points left after black points
    counter = 0

    if config.n_samples > 30:
        config.n_samples = 30
    # Create a new figure
    fig, ax = plt.subplots()
    # Plot the first few points in black
    for i in range(config.n_samples+1):
        ax.scatter(X[i, 0], X[i, 1], c='black', s=100, alpha=0.7)
    
    # Iterate over the remaining points and plot them in different colors
    for i in range(config.n_samples, len(X), number_point_list[counter]):
        for j in range(number_point_list[counter]-1):
            if i + j < len(X)-1:  # To ensure we don't go out of bounds
                plt.scatter(X[i+j, 0], X[i+j, 1], c=[colors(counter)], s=40, alpha=0.3)
        counter += 1  # Update the counter
        if counter>len(number_point_list)-1:
            break

    plt.title(title)
    plt.legend(loc='best')  # Adjusting legend location for best fit
    plt.show()

    # Save the plot to a BytesIO object and encode it as base64
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)  # Close the figure to free memory
    return html
# Example usage:
# Assuming X is your data, config.MMT_samples is 10, and the title is "2D Plot"
# plot_2D(X, 5, 20, config, "2D Plot")


# Convert SMILES to fingerprints
def smiles_to_fps(smiles_list):
    fps = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            fps.append(fp)
    return np.array(fps)



def plot_cluster_MMT(results_dict, config, mode, stoi, stoi_MF, itos, itos_MF):
    if "0" in results_dict:
        del results_dict["0"]
    if 0 in results_dict:
        del results_dict[0]

    transformed_list_MMT = [[key, [item[0] for item in value[0]]] for key, value in results_dict.items()]
    src_smi_MMT, key_len = list(results_dict.keys()), len(list(results_dict.keys()))
    # this leaves out the molecules with the highest tanimoto similarity to the target 
    # (assuming that it might be the correct molecules)
    combined_list_MMT = [item for sublist in transformed_list_MMT for item in sublist[1][1:]]  
    number_point_list_MMT = [len(sublist[1]) for sublist in transformed_list_MMT ]
    combined_list_new_MMT = src_smi_MMT+combined_list_MMT
    all_fps_MMT = smiles_to_fps(combined_list_new_MMT)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne_MMT = tsne.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap_MMT = umap_model.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca_MMT = pca.fit_transform(all_fps_MMT)


    # For testing purposes, let's use some dummy data

    html_TSNE = plot_2D(X_tsne_MMT, number_point_list_MMT, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D(X_umap_MMT, number_point_list_MMT, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D(X_pca_MMT, number_point_list_MMT, config, 'PCA Plot')
    
    return combined_list_MMT, html_TSNE, html_UMAP, html_PCA
    

def plot_cluster_MMT_2(config, gen_dict):

    transformed_list_MMT = [[key, value[0]] for key, value in gen_dict.items()]                           
    src_smi_MMT, key_len = list(gen_dict.keys()), len(list(gen_dict.keys()))
    config.n_samples = key_len                     
    # this leaves out the molecules with the highest tanimoto similarity to the target 
    # (assuming that it might be the correct molecules)
    combined_list_MMT = [item for sublist in transformed_list_MMT for item in sublist[1]] #[1:]]  
    number_point_list_MMT = [len(sublist[1]) for sublist in transformed_list_MMT ]
    combined_list_new_MMT = src_smi_MMT+combined_list_MMT
    all_fps_MMT = smiles_to_fps(combined_list_new_MMT)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne_MMT = tsne.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap_MMT = umap_model.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca_MMT = pca.fit_transform(all_fps_MMT)

    # For testing purposes, let's use some dummy data
    html_TSNE = plot_2D(X_tsne_MMT, number_point_list_MMT, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D(X_umap_MMT, number_point_list_MMT, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D(X_pca_MMT, number_point_list_MMT, config, 'PCA Plot')
    
    return combined_list_MMT, html_TSNE, html_UMAP, html_PCA        


def plot_cluster_MF(results_dict, config):
   
    transformed_list_MF = [[key, value] for key, value in results_dict.items()]
    src_smi_MF, key_len_MF = list(results_dict.keys()), len(list(results_dict.keys()))

    combined_list_MF = [item for sublist in transformed_list_MF for item in sublist[1][:]]
    number_point_list_MF = [len(sublist[1]) for sublist in transformed_list_MF ]
    combined_list_new_MF = src_smi_MF + combined_list_MF
    all_fps_MF = smiles_to_fps(combined_list_new_MF)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne_MF = tsne.fit_transform(all_fps_MF)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap_MF = umap_model.fit_transform(all_fps_MF)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca_MF = pca.fit_transform(all_fps_MF)


    # For testing purposes, let's use some dummy data

    html_TSNE = plot_2D(X_tsne_MF, number_point_list_MF, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D(X_umap_MF, number_point_list_MF, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D(X_pca_MF, number_point_list_MF, config, 'PCA Plot')
    
    return combined_list_MF, html_TSNE, html_UMAP, html_PCA
    
    

def plot_cluster_target(smi_list_target, smi_list_trainset, config):

    combined_list_new = smi_list_target+smi_list_trainset
    all_fps = smiles_to_fps(combined_list_new)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(all_fps)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap = umap_model.fit_transform(all_fps)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(all_fps)


    # For testing purposes, let's use some dummy data
    number_point_list = len(smi_list_target)
    html_TSNE = plot_2D_similarity(X_tsne, number_point_list, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D_similarity(X_umap, number_point_list, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D_similarity(X_pca, number_point_list, config, 'PCA Plot')
    
    return combined_list_new, html_TSNE, html_UMAP, html_PCA


def plot_2D_similarity(X, number_point_list, config, title):
    # Initialize variables
    colors = cm.get_cmap('tab20', config.data_size+1) # Adjust the colormap size based on the number of points left after black points
    counter = 0

    # Create a new figured
    fig, ax = plt.subplots()

    # Plot the first few points in black
    for i in range(number_point_list):
        ax.scatter(X[i, 0], X[i, 1], c='black', s=100, alpha=0.7)

    # Iterate over the remaining points and plot them in different colors
    for j in range(config.comparision_number):
        if i + j < len(X)-1:  # To ensure we don't go out of bounds
            plt.scatter(X[i+j, 0], X[i+j, 1], c="blue", s=40, alpha=0.3)

    plt.title(title)
    plt.legend(loc='best')  # Adjusting legend location for best fit
    plt.show()

    # Save the plot to a BytesIO object and encode it as base64
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)  # Close the figure to free memory

    # Return the HTML for the plot image
    return html



def run_cluster_comparision(config):
    # File path to the CSV file
    print("run_cluster_comparision")
    # train_file_path = '/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_train.csv'

    # Attempting to read the CSV file and select a random subset
    df_train = pd.read_csv(config.csv_train_path)

    # Filtering just the 'SMILES' column
    df_train = df_train[['SMILES']]

    # Selecting a random subset of 32 rows
    df_train = df_train.sample(n=config.comparision_number, random_state=1)
    smi_list_trainset = list(df_train["SMILES"])
    
    # File path to the CSV file
    target_file_path = config.csv_SMI_targets
    # Attempting to read the CSV file and select a random subset
    df_target = pd.read_csv(target_file_path)

    # Filtering just the 'SMILES' column
    df_target = df_target[['SMILES']]
    smi_list_targets = list(df_target["SMILES"])

    combined_list_new, html_TSNE, html_UMAP, html_PCA = plot_cluster_target(smi_list_targets, smi_list_trainset, config)
    
    return df_train, df_target, html_TSNE, html_UMAP, html_PCA
    