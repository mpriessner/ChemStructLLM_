import collections
import glob
import math
import ast
import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from rdkit.Chem import rdmolfiles
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, PandasTools, Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
from rdkit.Chem import SDMolSupplier, MolToSmiles
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from IPython.display import display, SVG
import utils_MMT.cosy_nmr_reconstruction_v15_4 as cnr
import utils_MMT.hsqc_nmr_reconstruction_v15_4 as hnr
import utils_MMT.helper_functions_pl_v15_4 as hf

from rdkit.Chem.PandasTools import LoadSDF


def load_acd_dataframe_from_file(acd_file_path):
    """
    Loads a dataframe from an ACD file.

    Parameters:
    acd_file_path (str): Path to the ACD file.

    Returns:
    DataFrame: Processed dataframe with deduplicated entries.
    """
    try:
        # Load data from the ACD file
        data = PandasTools.LoadSDF(acd_file_path)

        # Process the 'HSQC_13C-1H' column
        string_data = str(data.get('HSQC_13C-1H', [])[0])
        processed_data = [d.split(';') for d in string_data.split('\n') if d.strip()]
        processed_data_2 = [i[0].split("\t") for i in processed_data if i]

        if not processed_data_2:
            return pd.DataFrame()  # Return empty dataframe if data is missing or corrupt

        # Create dataframe from processed data
        df_acd = pd.DataFrame(processed_data_2[1:], columns=processed_data_2[0])

        # Determine direction based on 'F2 Atom' column
        direction = [-1 if "<" in i else 1 for i in df_acd["F2 Atom"]]
        df_acd["direction"] = direction

        # Deduplicate entries based on 'F2 (ppm)' and 'F1 (ppm)'
        df_acd_dedup = df_acd.drop_duplicates(
            subset=['F2 (ppm)', "F1 (ppm)"],
            keep='last').reset_index(drop=True)

        return df_acd_dedup
    except Exception as e:
        print(f"Error loading ACD file: {e}")
        return pd.DataFrame()  # Return empty dataframe in case of an error


# def plot_compare_scatter_plot_without_direction(df_orig, df_sim, name="Plot", transp=0.50):
#     """
#     Plots scatter plot from two dataframes on top of each other.

#     Parameters:
#     df_orig (DataFrame): Original dataframe.
#     df_sim (DataFrame): Simulated dataframe.
#     name (str): Title of the plot.
#     transp (float): Transparency of the scatter plot points.
#     """
#     plt.style.use('seaborn-whitegrid')

#     fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

#     # Plot experimental data
#     ax.scatter(df_orig['F2 (ppm)'].astype(float), df_orig['F1 (ppm)'].astype(float),
#                label="Experimental", alpha=transp, color="#1f77b4", s=50)

#     # Plot simulated data
#     ax.scatter(df_sim['F2 (ppm)'].astype(float), df_sim['F1 (ppm)'].astype(float),
#                label="Simulated", alpha=transp, color="#d62728", s=50)

#     ax.legend(fontsize=12)
#     ax.set_title(name, fontsize=16)
#     ax.set_xlim(0, 11)
#     ax.set_ylim(0, 200)
#     ax.invert_yaxis()
#     ax.invert_xaxis()
#     ax.set_xlabel('F2 (ppm)', fontsize=14)
#     ax.set_ylabel('F1 (ppm)', fontsize=14)
#     ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

#     plt.show()



# def plot_compare_scatter_plot(df_orig, df_sim, name="Plot", transp=0.50, style=["sim", "orig", "both"], direction=False):
#     """
#     Plots a scatter plot from two dataframes on top of each other. 
#     The 'direction' parameter can be used to distinguish between 
#     positive and negative intensities of the spectra.

#     Parameters:
#     df_orig (DataFrame): Original dataframe.
#     df_sim (DataFrame): Simulated dataframe.
#     name (str): Title of the plot.
#     transp (float): Transparency of the scatter plot points.
#     style (list): List of styles to include in the plot.
#     direction (bool): Flag to distinguish intensities direction.
#     """

#     # Setting phase column based on direction
#     phase_label = '+ve' if not direction else ['+ve', '-ve']
#     df_orig['Phase'] = phase_label + '_orig'
#     df_sim['Phase'] = phase_label + '_sim'

#     # Combining dataframes
#     df_combined = pd.concat([df_orig, df_sim], axis=0)

#     # Plotting
#     fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
#     if style == "both" or style in ["orig", "sim"]:
#         for phase in np.unique(df_combined['Phase']):
#             if style != "both" and not phase.endswith(style):
#                 continue
#             df_subset = df_combined[df_combined['Phase'] == phase]
#             ax.scatter(df_subset['F2 (ppm)'], df_subset['F1 (ppm)'], label=phase, alpha=transp)

#     ax.legend()
#     plt.title(name)
#     plt.xlim(0, 11)
#     plt.ylim(0, 200)
#     plt.gca().invert_yaxis()
#     plt.gca().invert_xaxis()
#     plt.grid()
#     plt.show()


# def plot_compare_scatter_plot(df_orig, df_sim, name="Plot", transp=0.50, style=["sim","orig","both"], direction=False):
#     """Plots scatter plot from two dataframes on top of each other. The direction ticker can be selected to 
#     distinguish between positive and negative intensities of the spectra (then displayed in 4 colours
#     TODO: the sim or orig alone plotting is not working yet
#     """

#     scatter_x_1 = list(np.array(df_orig['F2 (ppm)'].astype(float)))
#     scatter_y_1 = list(np.array(df_orig['F1 (ppm)'].astype(float)))
#     # intensity = np.array(df_orig['Intensity'].astype(float))
#     if direction:
#         df_orig['Phase'] = np.where(df_orig['Intensity']>=0, '+ve_orig', '-ve_orig')
#     else:
#         df_orig['Phase'] = np.where(df_orig['Intensity']>=0, '+ve_orig', '+ve_orig')
   

#     scatter_x_2 = list(np.array(df_sim['F2 (ppm)'].astype(float)))
#     scatter_y_2 = list(np.array(df_sim['F1 (ppm)'].astype(float)))
#     if direction:
#         df_sim['Phase'] = np.where(df_sim['direction']>=0, '+ve_sim', '-ve_sim')
#     else:
#         df_sim['Phase'] = np.where(df_sim['direction']>=0, '+ve_sim', '+ve_sim')

     
    
#     scatter_x = np.array(scatter_x_1 + scatter_x_2 )             
#     scatter_y = np.array(scatter_y_1 + scatter_y_2 ) 
#     df_orig = df_orig[['F2 (ppm)', 'F1 (ppm)', "Phase"]]
#     df_sim = df_sim[['F2 (ppm)', 'F1 (ppm)', "Phase"]]

#     df = pd.concat([df_orig, df_sim], axis=0) 
#     group = df['Phase']  

#     fig, ax = plt.subplots(figsize=(10,5), dpi=80)
#     if (style == "orig") or (style == "both"):
#         for g in np.unique(group):
#             i = np.where(group == g)
#             ax.scatter(scatter_x[i], scatter_y[i], label=g,  alpha=transp)

#     ax.legend()
#     plt.title(name)
#     plt.xlim(xmin=0,xmax=11)
#     plt.ylim(ymin=0,ymax=200)

#     plt.gca().invert_yaxis()
#     plt.gca().invert_xaxis()

#     plt.grid()
#     plt.show()


def get_similarity_comparison_variations(df_1, df_2, mode, sample_id, similarity_type=["euclidean", "cosine_similarity", "pearson_similarity"], error=["sum", "avg"], display_img=False):
    """
    Calculates various similarity measures for two HSQC dataframes.
    
    Parameters:
    df_1, df_2 (DataFrame): Input dataframes for similarity calculations.
    mode (str): Mode of similarity calculation.
    sample_id (str): Identifier for the sample.
    similarity_type (list): Types of similarity measures to calculate.
    error (list): Types of error calculation methods.
    display_img (bool): Flag to display a scatter plot.
    
    Returns:
    tuple: A tuple containing the list of similarity results and input dataframes.
    """

    input_dfs = {}
    similarity_results = []

    # Display scatter plot if requested
    if display_img:
        try:
            hf.plot_compare_scatter_plot(df_1, df_2, name=sample_id, transp=0.50, style="both", direction=False)
        except Exception:
            hf.plot_compare_scatter_plot_without_direction(df_1, df_2, name=sample_id, transp=0.50)

    # Define a list of modes to iterate over
    modes = ["min_sum_zero", "euc_dist_zero", "hung_dist_zero", "min_sum_trunc", "euc_dist_trunc", "hung_dist_trunc", "min_sum_nn", "euc_dist_nn", "hung_dist_nn"]

    # Iterate over modes and calculate similarities
    for current_mode in modes:
        if current_mode == mode:
            display_img = True

        similarity, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode=current_mode, similarity_type=similarity_type, error=error, assignment_plot=display_img)
        similarity_results.append(similarity)

        # Convert input lists to dataframes
        df1 = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
        df2 = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
        input_dfs[current_mode] = [df1, df2]

        display_img = False  # Reset the display flag for next iterations

    return similarity_results, input_dfs


def load_acd_dataframe_from_file(acd_file_path):
    """
    Loads and processes an ACD/Labs SDF file into a deduplicated pandas DataFrame.
    
    Parameters:
    acd_file_path (str): File path to the ACD/Labs SDF file.

    Returns:
    DataFrame: Processed DataFrame with deduplicated entries.
    """
    # Load SDF file into a DataFrame
    data = PandasTools.LoadSDF(acd_file_path)

    # Process HSQC_13C-1H data
    string_data = str(data['HSQC_13C-1H'][0])
    processed_data = [d.split(';') for d in string_data.split('\n')]
    processed_data_2 = [item[0].split("\t") for item in processed_data]

    # Create DataFrame from processed data
    df_acd = pd.DataFrame(processed_data_2[1:], columns=processed_data_2[0])

    # Determine direction based on 'F2 Atom' column
    df_acd['direction'] = df_acd['F2 Atom'].apply(lambda x: -1 if "<" in x else 1)

    # Deduplicate entries based on 'F2 (ppm)' and 'F1 (ppm)'
    df_acd_dedup = df_acd.drop_duplicates(subset=['F2 (ppm)', 'F1 (ppm)'], keep='last').reset_index(drop=True)

    return df_acd_dedup



def load_real_dataframe_from_file(real_file_path):
    """
    Loads a real dataset from a file and processes it into a pandas DataFrame.

    Parameters:
    real_file_path (str): File path to the real dataset.

    Returns:
    DataFrame: Processed DataFrame with renamed columns.
    """
    try:
        # Attempt to read the file with flexible separators (tab or whitespace)
        df_real = pd.read_csv(real_file_path, sep=r'\t|\s+', engine='python')

        # Rename columns for consistency
        df_real = df_real.rename(columns={"F2ppm": "F2 (ppm)", "F1ppm": "F1 (ppm)"})
        return df_real

    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def load_mnova_dataframe_from_file(file_path):
    """
    Loads a dataset from an MNOVA file and processes it into a pandas DataFrame.

    Parameters:
    file_path (str): File path to the MNOVA dataset.

    Returns:
    DataFrame: Processed DataFrame with specified column names.
    """
    try:
        # Read the file with flexible separators (tab or whitespace) and specified column names
        df_mnova = pd.read_csv(file_path, sep=r'\t|\s+', engine='python', names=["id", "F2 (ppm)", "F1 (ppm)", 'Intensity'])
        return df_mnova

    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def load_HSQC_dataframe_from_file(ml_file_path):
    """
    Loads HSQC data from an ML SDF file into a DataFrame.

    Parameters:
    ml_file_path (str): Path to the ML SDF file.

    Returns:
    DataFrame: A DataFrame containing the processed HSQC data.
    """

    try:
        # Load data from SDF file
        data = PandasTools.LoadSDF(ml_file_path)
        str_shifts = data["averaged_NMR_shifts"].item()
        boltzman_avg_shifts_corr_2 = [float(i) for i in str_shifts.split()]

        # Process molecule data
        sym_dupl_lists, all_split_positions, mol, compound_path = hnr.run_chiral_and_symmetry_finder(compound_path=ml_file_path)
        atom_list, connectivity_list, docline_list, name, mol = hnr.get_molecule_data(ml_file_path)
        c_h_connectivity_dict = hnr.get_c_h_connectivity(connectivity_list, atom_list)

        # Select and deduplicate shifts
        shifts = hnr.selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
        shifts = hnr.perform_deduplication_if_symmetric(shifts, sym_dupl_lists)

        # Generate DataFrame from shifts
        df_ml = hnr.generate_dft_dataframe(shifts)
        return df_ml

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None


def load_COSY_dataframe_from_file(ml_file_path):
    """
    Loads COSY data from an ML SDF file into a DataFrame.

    Parameters:
    ml_file_path (str): Path to the ML SDF file.

    Returns:
    DataFrame: A DataFrame containing the processed COSY data.
    """

    try:
        # Load the molecule from the file
        mol = SDMolSupplier(ml_file_path)[0]

        # Enumerate stereo isomers and convert to SMILES
        isomers = tuple(EnumerateStereoisomers(mol))
        stereo_smi = MolToSmiles(isomers[0], isomericSmiles=True)

        # Extract averaged NMR shifts and sample ID
        averaged_nmr_shifts = mol.GetProp('averaged_NMR_shifts')
        sample_shifts = list(map(float, averaged_nmr_shifts.split()))
        file_name = os.path.basename(ml_file_path)
        sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

        # Find chiral centers and carbon atoms
        chiral_centers = cnr.find_chiral_centers(mol)
        carbon_dict = cnr.find_carbons_with_relevant_neighbors(mol)
        heavy_atom_dict = cnr.find_heavy_atoms_with_hydrogens(mol)

        # Process shifts and detect symmetric positions
        heavy_atom_hydrogen_shift_dict = cnr.extract_symmetric_hydrogen_shifts(sample_shifts, heavy_atom_dict)
        sym_dupl_lists = cnr.find_symmetric_positions(stereo_smi)
        sym_dupl_lists = [positions for positions in sym_dupl_lists if all(cnr.has_hydrogens(mol, idx) for idx in positions)]

        # Average shifts and update dictionary
        averaged_shifts = cnr.average_shifts(heavy_atom_hydrogen_shift_dict, sym_dupl_lists)
        updated_heavy_atom_hydrogen_shift_dict = cnr.update_shifts_with_averaged(heavy_atom_hydrogen_shift_dict, averaged_shifts)

        # Process COSY shifts and generate DataFrame
        COSY_shifts = cnr.plot_and_save_cosy_spectrum_with_zoom_no_duplicates(updated_heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers, plot=False, xlim=None, ylim=None)
        COSY_shifts = sorted(COSY_shifts, key=lambda x: x[0])
        df_COSY = cnr.generate_COSY_dataframe(COSY_shifts)
        
        return df_COSY

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None

def load_shifts_from_dp_sdf_file(sdf_file_dp):
    """
    Loads chemical shifts from a given SDF file generated by Deep Picker (DP).

    Parameters:
    sdf_file_dp (str): Path to the SDF file.

    Returns:
    list: A list of chemical shifts, or None if an error occurs.
    """

    try:
        # Load the SDF file into a DataFrame
        data = LoadSDF(sdf_file_dp)

        # Extract and parse the 'averaged_NMR_shifts' property
        chemical_shifts_str = data["averaged_NMR_shifts"][0]
        chemical_shifts = ast.literal_eval(chemical_shifts_str)

        return chemical_shifts

    except Exception as e:
        print(f"Error loading chemical shifts from SDF file: {e}")
        return None

def load_dft_dataframe_from_file(dft_file_path):
    """Load a DFT (density functional theory) chemical shift file in SD format to a pandas DataFrame.
    Args:
        dft_file_path (str): The path to the DFT file in SD format.
    Returns:
        pandas.DataFrame: A DataFrame containing the chemical shifts and corresponding atomic positions for the DFT file.
    """
    boltzman_avg_shifts_corr_2 = load_shifts_from_dp_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = hnr.run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = hnr.get_molecule_data(dft_file_path)
    c_h_connectivity_dict = hnr.get_c_h_connectivity(connectivity_list, atom_list)
    shifts = hnr.selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = hnr.perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_dft = hnr.generate_dft_dataframe(shifts)
    dft_num_peaks = len(df_dft)
    return df_dft

def load_real_df_from_txt_path(path_txt):
    """
    Prepares a DataFrame from a TXT file for plotting real NMR data.

    Parameters:
    path_txt (str): Path to the TXT file.

    Returns:
    tuple: A tuple containing the DataFrame and the name extracted from the file path.
    """

    try:
        # Attempt to read the file with multiple separators
        df_real = pd.read_csv(path_txt, sep="\t|\s+", engine='python')

    except Exception as e:
        print(f"Error reading file with multiple separators. Trying a single tab separator: {e}")
        try:
            df_real = pd.read_csv(path_txt, sep="\t")
        except Exception as e:
            print(f"Failed to load file: {e}")
            return None, None

    # Rename columns for consistency
    if 'F2ppm' in df_real.columns and 'F1ppm' in df_real.columns:
        df_real.rename(columns={'F2ppm': 'F2 (ppm)', 'F1ppm': 'F1 (ppm)'}, inplace=True)
    else:
        print("Expected columns 'F2ppm' and 'F1ppm' not found in file.")
        return None, None

    # Extract the file name for naming
    name = os.path.basename(path_txt).split(".")[0]

    return df_real, name


def similarity_calculations(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
                            similarity_type=["euclidean","cosine_similarity"],  \
                            error=["sum","avg"], \
                            assignment_plot = True):
    """This function calculates the cosine similarity of xy of two spectra provided by dataframes
    by first normalizing it and then choosing one of two modes
    min_sum: takes the minimum sum of x + y as a sorting criteria
    euc_dist: compares every point with each other point of the spectra and matches them to minimize the error"""

    h_dim_1 = list(np.array(df_1['F2 (ppm)'].astype(float))/10-0.5)
    c_dim_1 = list(np.array(df_1['F1 (ppm)'].astype(float))/200-0.5)

    h_dim_2 = list(np.array(df_2['F2 (ppm)'].astype(float))/10-0.5)
    c_dim_2 = list(np.array(df_2['F1 (ppm)'].astype(float))/200-0.5)

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
        input_list_1, input_list_2 = min_sum_nn(input_list_1,input_list_2)

    elif mode == "euc_dist_zero":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num)

    elif mode == "euc_dist_trunc":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num)
        input_list_1, input_list_2 = filter_out_zeros(input_list_1, input_list_2)

    elif mode == "euc_dist_nn":
        input_list_1, input_list_2 = euclidean_distance_nn(input_list_1,input_list_2)

    elif mode == "hung_dist_zero":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)

    elif mode == "hung_dist_trunc":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num)
        input_list_1, input_list_2 = filter_out_zeros(input_list_1, input_list_2)
        input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)

    elif mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2 = hungarian_advanced_euc(input_list_1,input_list_2)

    if similarity_type == "cosine_similarity":
        from scipy.spatial import distance
        list_points_1 = np.array(input_list_1, dtype=object)
        list_points_2 = np.array(input_list_2, dtype=object)
        Aflat = np.hstack(list_points_1)
        Bflat = np.hstack(list_points_2)
        # Aflat = Aflat - Aflat.mean()
        # Bflat = Bflat - Bflat.mean()
        cos_sim = 1 - distance.cosine(Aflat, Bflat)
        if assignment_plot == True:
            plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, cos_sim)
            pass
        return cos_sim, np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()

    elif similarity_type == "euclidean":
        sum_dist = 0
        max_dist = 0
        for sample_1, sample_2 in zip(input_list_1, input_list_2):
            from scipy.spatial import distance
            dst = distance.euclidean(sample_1, sample_2)
            sum_dist+=dst
            max_dist = max(dst, max_dist)
        if error=="avg":
            similarity_type = similarity_type + "_" + error
            ############# new addition #############
            if not "trunc" in mode:
                avg_dist = sum_dist/max(len(input_list_1),len(input_list_2))
            elif  "trunc" in mode:
                avg_dist = sum_dist/min(len(input_list_1),len(input_list_2))
            if assignment_plot == True:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, avg_dist)
                pass
            return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()
        elif error=="sum":
            similarity_type = similarity_type + "_" + error
            sum_error = np.array(sum_dist)
            if assignment_plot == True:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, sum_error)
                pass
            return sum_error, np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()



def padding_to_max_length(input_list_1, input_list_2, dim=2):
    """
    Pads the shorter of two input lists with zeros to match the length of the longer list.

    Parameters:
    input_list_1, input_list_2 (numpy array): Input arrays to be padded.
    dim (int): Dimension for padding.

    Returns:
    tuple: Padded input lists and the number of padding rows added.
    """

    len_1, len_2 = len(input_list_1), len(input_list_2)
    pad_num = abs(len_1 - len_2)  # Calculate the difference in lengths

    if len_1 < len_2:
        # Pad input_list_1 if it's shorter
        padding_matrix = np.zeros((pad_num, dim))
        input_list_1 = np.concatenate((input_list_1, padding_matrix), axis=0)
    elif len_2 < len_1:
        # Pad input_list_2 if it's shorter
        padding_matrix = np.zeros((pad_num, dim))
        input_list_2 = np.concatenate((input_list_2, padding_matrix), axis=0)

    return input_list_1, input_list_2, pad_num


def plot_assignment_points(input_list_orig, input_list_sim, mode, similarity_type, error):
    """
    Plots experimental and simulated data points with assignment lines.

    Parameters:
    input_list_orig (numpy array): Original (experimental) data points.
    input_list_sim (numpy array): Simulated data points.
    mode (str): The mode of similarity calculation.
    similarity_type (str): Type of similarity measure used.
    error (float): Calculated error or similarity score.
    """

    # Assuming assignments are made in order
    assignment = list(range(len(input_list_orig)))

    # Plot settings
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Plotting the data points
    ax.plot(input_list_orig[:, 0], input_list_orig[:, 1], 'bo', markersize=10, label='Experimental')
    ax.plot(input_list_sim[:, 0], input_list_sim[:, 1], 'rs', markersize=7, label='Simulated')

    # Drawing assignment lines
    for p in range(len(input_list_orig)):
        ax.plot([input_list_orig[p, 0], input_list_sim[assignment[p], 0]], 
                [input_list_orig[p, 1], input_list_sim[assignment[p], 1]], 
                'k--', alpha=0.5)

    # Setting titles and labels
    title_str = f'{mode}_{similarity_type}: {round(error, 3)}'
    ax.set_title(title_str, fontsize=16)
    ax.set_xlabel('Normalized 1H Shifts', fontsize=14)
    ax.set_ylabel('Normalized 13C Shifts', fontsize=14)

    # Inverting axes
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Adding legend
    ax.legend(fontsize=12)

    plt.show()



def euclidean_distance_zero_padded(input_list_1, input_list_2, num_pad):
    """
    Aligns points from two lists based on Euclidean distance, matching remaining ones with zero padding.

    Parameters:
    input_list_1 (numpy array): First list of points.
    input_list_2 (numpy array): Second list of points.
    num_pad (int): Number of padding zeros.

    Returns:
    Tuple: Aligned dataset arrays.
    """

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


    return np.array(dataset_1), np.array(dataset_2)


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


def filter_out_zeros(input_list_1, input_list_2):
    """
    Filters out pairs of elements from input_list_1 and input_list_2 where both
    elements in the pair are close to zero. Returns two new lists containing the
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
        if (abs(i[0])+abs(i[1]))<0.001 or (abs(j[0]) + abs(j[1]))<0.001:
            continue
        else:
            new_list_1.append(list(i))
            new_list_2.append(list(j))
    new_list_1 = np.array(new_list_1)
    new_list_2 = np.array(new_list_2)
    return new_list_1, new_list_2




def min_sum_nn(input_list_1, input_list_2):
    """
    Aligns two lists of points using a combination of minimum sum and Euclidean distance.

    Parameters:
    input_list_1 (numpy array): First list of points.
    input_list_2 (numpy array): Second list of points.

    Returns:
    Tuple: Arrays of aligned points from input_list_1 and input_list_2.
    """
    input_list_1 = np.array(sorted(input_list_1, key = lambda x: -(x[0]+x[1])))
    input_list_2 = np.array(sorted(input_list_2, key = lambda x: -(x[0]+x[1])))
    min_length = min(len(input_list_1),len(input_list_2))
    input_list_1_part_1 = input_list_1[:min_length]
    input_list_2_part_1 = input_list_2[:min_length]
    input_list_1_align = list(input_list_1_part_1)
    input_list_2_align = list(input_list_2_part_1)

    if len(input_list_2) > len(input_list_1):
        input_list_2_part_2 = input_list_2[min_length:]

        input_list_1_part_2, input_list_2_part_2  = euclidean_distance_uneven(input_list_1,input_list_2_part_2)
        input_list_1_align.extend(list(input_list_1_part_2))
        input_list_2_align.extend(list(input_list_2_part_2))


    elif len(input_list_2) < len(input_list_1):
        input_list_1_part_2 = input_list_1[min_length:]

        input_list_1_part_2, input_list_2_part_2 = euclidean_distance_uneven(input_list_1_part_2, input_list_2)
        input_list_1_align.extend(list(input_list_1_part_2))
        input_list_2_align.extend(list(input_list_2_part_2))


    return np.array(input_list_1_align), np.array(input_list_2_align)



def euclidean_distance_nn(input_list_1,input_list_2):
    """The Euclidean-Distance-Advanced first matches each point from one set to the second and then picks the
    remaining points that were not matched because of a mismatch of number of points and matches them again
    to the shorter set. """
    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
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
    input_list_1_align, input_list_2_align = euclidean_distance_uneven(input_list_1,input_list_2)

    # # Select the remaining points that have not been matched in the first round for second alignment
    input_list_1_align_part_2 = []
    input_list_2_align_part_2 = []
    if len(input_list_1)< len(input_list_2):
        for i in input_list_2 :
            if i not in input_list_2_align:
                input_list_2_align_part_2.append(i)

        # match them again with full number of points from other set
        input_list_1_align_part_2, input_list_2_align_part_2 = euclidean_distance_uneven(input_list_1,input_list_2_align_part_2)

    elif len(input_list_1)> len(input_list_2):
        for i in input_list_1:
            if i not in input_list_1_align:
                input_list_1_align_part_2.append(i)

        # match them again with full number of points from other set
        input_list_1_align_part_2, input_list_2_align_part_2 = euclidean_distance_uneven(input_list_1_align_part_2,input_list_2)

    # Combine both to final list
    input_list_1_align = list(input_list_1_align)
    input_list_2_align = list(input_list_2_align)
    input_list_1_align.extend(list(input_list_1_align_part_2))
    input_list_2_align.extend(list(input_list_2_align_part_2))
    return np.array(input_list_1_align), np.array(input_list_2_align)



def euclidean_distance_uneven(input_list_1,input_list_2):
    """This function aligns the closest points with each other based on euclidean distance
    and matches the remaining ones with the zero padding"""

    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
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

def hungarian_advanced_euc(input_list_1,input_list_2):
    #Hungarian advanced
    input_list_1_euc,input_list_2_euc = euclidean_distance_nn(input_list_1,input_list_2)
    input_list_1_euc_hung,input_list_2_euc_hung = hungarian_zero_padded(input_list_1_euc,input_list_2_euc)
    return np.array(input_list_1_euc_hung),np.array(input_list_2_euc_hung)

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