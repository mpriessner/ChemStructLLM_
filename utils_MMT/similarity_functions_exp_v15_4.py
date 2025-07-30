# Standard library imports
import os
import ast
import collections

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from IPython.display import display, SVG

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, SDMolSupplier, MolToSmiles, AddHs, PandasTools
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem import rdmolfiles

#import sys
#sys.path.append("/projects/cc/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/utils_MMT")
import utils_MMT.cosy_nmr_reconstruction_v15_4 as cnr
import utils_MMT.hsqc_nmr_reconstruction_v15_4 as hnr
import utils_MMT.helper_functions_pl_v15_4 as hf
#import utils_MMT.data_generation_v15_4 as dg
import utils_MMT.extracting_data_KG_v15_4 as ed




############################################################################################
################################## Similartiy new ##########################################
############################################################################################
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def normalize_spectra(df_1, df_2, spectrum_type):
    """
    Normalizes spectral data based on spectrum type.
    """
    # First ensure numeric conversion
    if spectrum_type == 'HSQC':
        # F2 is proton dimension (1H)
        h_dim_1 = pd.to_numeric(df_1['F2 (ppm)'], errors='coerce').values/10-0.5
        # F1 is carbon dimension (13C)
        c_dim_1 = pd.to_numeric(df_1['F1 (ppm)'], errors='coerce').values/200-0.5
        h_dim_2 = pd.to_numeric(df_2['F2 (ppm)'], errors='coerce').values/10-0.5
        c_dim_2 = pd.to_numeric(df_2['F1 (ppm)'], errors='coerce').values/200-0.5
    
    elif spectrum_type == 'COSY':
        h_dim_1 = pd.to_numeric(df_1['F2 (ppm)'], errors='coerce').values/10-0.5
        c_dim_1 = pd.to_numeric(df_1['F1 (ppm)'], errors='coerce').values/10-0.5
        h_dim_2 = pd.to_numeric(df_2['F2 (ppm)'], errors='coerce').values/10-0.5
        c_dim_2 = pd.to_numeric(df_2['F1 (ppm)'], errors='coerce').values/10-0.5
    
    elif spectrum_type == '13C':
        h_dim_1 = pd.to_numeric(df_1['shifts'], errors='coerce').values/200-0.5
        c_dim_1 = np.ones(len(df_1['shifts']))
        h_dim_2 = pd.to_numeric(df_2['shifts'], errors='coerce').values/200-0.5
        c_dim_2 = np.ones(len(df_2['shifts']))
    
    elif spectrum_type == '1H':
        h_dim_1 = pd.to_numeric(df_1['shifts'], errors='coerce').values/10-0.5
        c_dim_1 = np.ones(len(df_1['shifts']))
        h_dim_2 = pd.to_numeric(df_2['shifts'], errors='coerce').values/10-0.5
        c_dim_2 = np.ones(len(df_2['shifts']))
    
    else:
        raise ValueError(f"Unsupported spectrum type: {spectrum_type}")

    # Convert to lists and ensure atom_index is present
    input_list_1 = np.array([list(h_dim_1), list(c_dim_1), df_1.atom_index.values]).transpose()
    input_list_2 = np.array([list(h_dim_2), list(c_dim_2), df_2.atom_index.values]).transpose()
    
    return input_list_1, input_list_2

def euclidean_distance_all(input_list_1, input_list_2):
    """
    Matches points using Euclidean distance, allowing multiple matches.
    """
    coords_1, atom_indices_1 = input_list_1[:, :2].astype(float), input_list_1[:, 2]
    coords_2, atom_indices_2 = input_list_2[:, :2].astype(float), input_list_2[:, 2]
    
    distances = cdist(coords_1, coords_2)
    matched_pairs = []
    errors = []
    matching_indices_1 = []
    matching_indices_2 = []
    
    # First pass: match each point to its closest available point
    used_indices_2 = set()
    for i in range(len(coords_1)):
        min_dist_idx = np.argmin(distances[i])
        matched_pairs.append((i, min_dist_idx))
        errors.append(distances[i][min_dist_idx])
        matching_indices_1.append(atom_indices_1[i])
        matching_indices_2.append(atom_indices_2[min_dist_idx])
        used_indices_2.add(min_dist_idx)
    
    # Second pass: match remaining points in list 2
    for j in range(len(coords_2)):
        if j not in used_indices_2:
            min_dist_idx = np.argmin(distances[:, j])
            matched_pairs.append((min_dist_idx, j))
            errors.append(distances[min_dist_idx][j])
            matching_indices_1.append(atom_indices_1[min_dist_idx])
            matching_indices_2.append(atom_indices_2[j])
    
    output_1 = []
    output_2 = []
    for i, j in matched_pairs:
        output_1.append(coords_1[i])
        output_2.append(coords_2[j])
    
    return np.array(output_1), np.array(output_2), np.array(errors), np.array(matching_indices_1), np.array(matching_indices_2)

def euclidean_distance_all(input_list_1, input_list_2):
    """
    Matches points using Euclidean distance, first handling the shorter list
    then matching remaining points from the longer list.
    
    Parameters:
    input_list_1, input_list_2 (numpy.array): Input arrays with coordinates and atom indices
    
    Returns:
    tuple: Matched arrays, errors, and matching indices
    """
    coords_1, atom_indices_1 = input_list_1[:, :2].astype(float), input_list_1[:, 2]
    coords_2, atom_indices_2 = input_list_2[:, :2].astype(float), input_list_2[:, 2]
    
    # Determine which list is shorter
    if len(coords_1) <= len(coords_2):
        shorter_coords = coords_1
        shorter_indices = atom_indices_1
        longer_coords = coords_2
        longer_indices = atom_indices_2
        shorter_is_first = True
    else:
        shorter_coords = coords_2
        shorter_indices = atom_indices_2
        longer_coords = coords_1
        longer_indices = atom_indices_1
        shorter_is_first = False
    
    # Calculate distance matrix
    distances = cdist(shorter_coords, longer_coords)
    
    output_shorter = []
    output_longer = []
    errors = []
    matching_indices_shorter = []
    matching_indices_longer = []
    used_longer_indices = set()
    
    # First pass: match each point in shorter list to its closest point in longer list
    for i in range(len(shorter_coords)):
        min_dist_idx = np.argmin(distances[i])
        output_shorter.append(shorter_coords[i])
        output_longer.append(longer_coords[min_dist_idx])
        errors.append(distances[i][min_dist_idx])
        matching_indices_shorter.append(shorter_indices[i])
        matching_indices_longer.append(longer_indices[min_dist_idx])
        used_longer_indices.add(min_dist_idx)
    
    # Second pass: match remaining points from longer list to their closest matches
    remaining_indices = [i for i in range(len(longer_coords)) if i not in used_longer_indices]
    for i in remaining_indices:
        # Calculate distances to all points in shorter list
        point_distances = cdist([longer_coords[i]], shorter_coords)[0]
        min_dist_idx = np.argmin(point_distances)
        
        output_shorter.append(shorter_coords[min_dist_idx])
        output_longer.append(longer_coords[i])
        errors.append(point_distances[min_dist_idx])
        matching_indices_shorter.append(shorter_indices[min_dist_idx])
        matching_indices_longer.append(longer_indices[i])
    
    # Reorder outputs if needed
    if shorter_is_first:
        return (np.array(output_shorter), np.array(output_longer), 
                np.array(errors), 
                np.array(matching_indices_shorter), 
                np.array(matching_indices_longer))
    else:
        return (np.array(output_longer), np.array(output_shorter), 
                np.array(errors), 
                np.array(matching_indices_longer), 
                np.array(matching_indices_shorter))

def hungarian_advanced_euc(input_list_1, input_list_2):
    """
    Matches points using Hungarian algorithm after initial Euclidean distance matching.
    """
    coords_1, atom_indices_1 = input_list_1[:, :2].astype(float), input_list_1[:, 2]
    coords_2, atom_indices_2 = input_list_2[:, :2].astype(float), input_list_2[:, 2]
    
    # Add small random noise
    coords_1 += np.random.random(coords_1.shape) * 1e-10
    coords_2 += np.random.random(coords_2.shape) * 1e-10
    
    cost_matrix = cdist(coords_1, coords_2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    errors = [cost_matrix[i][j] for i, j in zip(row_ind, col_ind)]
    
    # Only return the coordinates for matched points
    output_1 = coords_1[row_ind]
    output_2 = coords_2[col_ind]
    matching_indices_1 = atom_indices_1[row_ind]
    matching_indices_2 = atom_indices_2[col_ind]
    
    return output_1, output_2, np.array(errors), matching_indices_1, matching_indices_2

def denormalize_output(df_out, spectrum_type):
    """
    Denormalizes the output dataframe based on spectrum type.
    """
    if spectrum_type == 'HSQC' or spectrum_type == 'COSY':
        df_out['F2 (ppm)'] = pd.to_numeric(df_out['F2 (ppm)'], errors='coerce')
        df_out['F1 (ppm)'] = pd.to_numeric(df_out['F1 (ppm)'], errors='coerce')
        # For HSQC: F2 is proton (1H), F1 is carbon (13C)
        # For COSY: both F1 and F2 are proton (1H)
        df_out['F2 (ppm)'] = (df_out['F2 (ppm)'] + 0.5) * 10  # Always proton dimension
        df_out['F1 (ppm)'] = (df_out['F1 (ppm)'] + 0.5) * (200 if spectrum_type == 'HSQC' else 10)
    elif spectrum_type == '13C':
        df_out['shifts'] = pd.to_numeric(df_out['shifts'], errors='coerce')
        df_out['shifts'] = (df_out['shifts'] + 0.5) * 200
    elif spectrum_type == '1H':
        df_out['shifts'] = pd.to_numeric(df_out['shifts'], errors='coerce')
        df_out['shifts'] = (df_out['shifts'] + 0.5) * 10
    
    return df_out

def unified_similarity_calculation(df_1, df_2, spectrum_type, method='hung_dist_nn', error_type='sum'):
    """
    Calculates similarity between two spectra using specified method.
    """
    # Ensure we have copies to avoid modifying original dataframes
    df_1 = df_1.copy()
    df_2 = df_2.copy()
    print(f"\nSpectrum df_1: {df_1}")
    # Normalize input data
    input_list_1, input_list_2 = normalize_spectra(df_1, df_2, spectrum_type)
    print(f"\nSpectrum input_list_1: {input_list_1}")

    # Apply selected matching method
    if method == 'hung_dist_nn':
        matched_1, matched_2, errors, indices_1, indices_2 = hungarian_advanced_euc(input_list_1, input_list_2)
    elif method == 'euc_dist_all':
        matched_1, matched_2, errors, indices_1, indices_2 = euclidean_distance_all(input_list_1, input_list_2)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Add additional columns
    print(f"\nSpectrum type: {spectrum_type}")
    print("Adding errors to dataframes:")
    for idx, (err, idx1, idx2) in enumerate(zip(errors, indices_1, indices_2)):
        print(f"Peak {idx}:")
        print(f"  Error: {err}")
        print(f"  Matching indices: {idx1} -> {idx2}")

    # Calculate final error
    final_error = np.mean(errors) if error_type == 'avg' else np.sum(errors)
    
    # Prepare column names based on spectrum type
    columns = {
        'HSQC': ['F2 (ppm)', 'F1 (ppm)'],
        'COSY': ['F2 (ppm)', 'F1 (ppm)'],
        '13C': ['shifts', 'intensity'],
        '1H': ['shifts', 'intensity']
    }
    
    col_names = columns[spectrum_type]
    
    # Create output dataframes with just the coordinate columns
    df_1_out = pd.DataFrame(matched_1, columns=col_names)
    df_2_out = pd.DataFrame(matched_2, columns=col_names)
    
    # Add additional columns
    df_1_out['atom_index'] = indices_1
    df_2_out['atom_index'] = indices_2
    df_1_out['Error'] = errors
    df_2_out['Error'] = errors
    df_1_out['Self_Index'] = indices_1
    df_1_out['Matching_Index'] = indices_2
    df_2_out['Self_Index'] = indices_2
    df_2_out['Matching_Index'] = indices_1
    
    # Denormalize outputs
    print(f"\denormalize_output : {df_1_out}")

    df_1_out = denormalize_output(df_1_out, spectrum_type)
    df_2_out = denormalize_output(df_2_out, spectrum_type)
    
    return final_error, df_1_out, df_2_out

############################################################################################
####################################### HSQC ###############################################
############################################################################################

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
        atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(ml_file_path)
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
        stereo_smi = MolToSmiles(isomers[0],  isomericSmiles=False, canonical=True)  

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
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(dft_file_path)
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


def similarity_calculations_HSQC(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
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
    
    # Include original indices
    input_list_1 = np.array([h_dim_1, c_dim_1, df_1.atom_index]).transpose()
    input_list_2 = np.array([h_dim_2, c_dim_2, df_2.atom_index]).transpose()
    #sorting by sum of x,y high to low
    ######### Do alignment algorithm instead
    #import IPython; IPython.embed();
    if mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2, errors, matching_indices_1, matching_indices_2 = hungarian_advanced_euc(input_list_1,input_list_2)
        from scipy.spatial import distance
        #errors_ = np.array([distance.euclidean(a, b) for a, b in zip(input_list_1, input_list_2)])
        #print("in HSQC Sim function")
        #import IPython; IPython.embed();
    elif mode == "hung_dist_zero":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)

    # elif mode == "hung_dist_trunc":
    #     input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
    #     input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num)
    #     input_list_1, input_list_2 = filter_out_zeros(input_list_1, input_list_2)
    #     input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)

        
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
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])

            # Convert 'F2 (ppm)' and 'F1 (ppm)' to numeric for df_1_out
            df_1_out['F2 (ppm)'] = pd.to_numeric(df_1_out['F2 (ppm)'], errors='coerce')
            df_1_out['F1 (ppm)'] = pd.to_numeric(df_1_out['F1 (ppm)'], errors='coerce')

            # Apply transformations for df_1_out
            df_1_out['F2 (ppm)'] = (df_1_out['F2 (ppm)'] + 0.5) * 10
            df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 200

            # Convert 'F2 (ppm)' and 'F1 (ppm)' to numeric for df_2_out
            df_2_out['F2 (ppm)'] = pd.to_numeric(df_2_out['F2 (ppm)'], errors='coerce')
            df_2_out['F1 (ppm)'] = pd.to_numeric(df_2_out['F1 (ppm)'], errors='coerce')

            # Apply transformations for df_2_out
            df_2_out['F2 (ppm)'] = (df_2_out['F2 (ppm)'] + 0.5) * 10
            df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 200

            
            if assignment_plot:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, avg_dist)

            #return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()
            return np.array(avg_dist), df_1_out, df_2_out
        
        elif error=="sum":
            similarity_type = similarity_type + "_" + error
            sum_error = np.array(sum_dist)
            #import IPython; IPython.embed();
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            
            # Convert 'F2 (ppm)' and 'F1 (ppm)' to numeric for df_1_out
            df_1_out['F2 (ppm)'] = pd.to_numeric(df_1_out['F2 (ppm)'], errors='coerce')
            df_1_out['F1 (ppm)'] = pd.to_numeric(df_1_out['F1 (ppm)'], errors='coerce')

            # Apply transformations for df_1_out
            df_1_out['F2 (ppm)'] = (df_1_out['F2 (ppm)'] + 0.5) * 10
            df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 200

            # Convert 'F2 (ppm)' and 'F1 (ppm)' to numeric for df_2_out
            df_2_out['F2 (ppm)'] = pd.to_numeric(df_2_out['F2 (ppm)'], errors='coerce')
            df_2_out['F1 (ppm)'] = pd.to_numeric(df_2_out['F1 (ppm)'], errors='coerce')

            # Apply transformations for df_2_out
            df_2_out['F2 (ppm)'] = (df_2_out['F2 (ppm)'] + 0.5) * 10
            df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 200

            
            if assignment_plot == True:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, float(sum_error))
                pass
           
            return sum_error,  df_1_out, df_2_out

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
        padding_matrix[:,-1] = -1  # Set the third column (index) to -1       
        input_list_1 = np.concatenate((input_list_1, padding_matrix), axis=0)
    elif len_2 < len_1:
        # Pad input_list_2 if it's shorter
        padding_matrix = np.zeros((pad_num, dim))
        padding_matrix[:, -1] = -1  # Set the third column (index) to -1        
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
    input_list_1 (numpy array): First list of points with shape (n, 3) where the last column is the atom index.
    input_list_2 (numpy array): Second list of points with shape (m, 3) where the last column is the atom index.
    num_pad (int): Number of padding zeros.
    Returns:
    Tuple: Aligned dataset arrays, errors, and matching indices.
    """
    # Separate coordinates and atom indices
    coords_1, atom_indices_1 = input_list_1[:, :2], input_list_1[:, 2]
    coords_2, atom_indices_2 = input_list_2[:, :2], input_list_2[:, 2]

    # Add small random numbers to avoid exact duplicates
    coords_1 += np.random.random(coords_1.shape) * 1e-10
    coords_2 += np.random.random(coords_2.shape) * 1e-10

    # Calculate euclidean distances
    #distances = cdist(coords_1, coords_2)
    # calculate euclidean distance
    result = []
    for j, a_i1 in zip(coords_1, atom_indices_1):
        for i, a_i2 in zip(coords_2, atom_indices_2):
            dst = distance.euclidean(i, j)
            result.append([list(i),list(j), a_i2, a_i1, dst])

    # sort it for the lowest euclidean distance
    result = sorted(result, key=lambda x:x[-1], reverse=False)

    # This aligns the closest points with each other
    # and compares the remaining with the zero padding
    dataset_1 = []
    dataset_2 = []
    errors = []
    matching_indices_1 = []
    matching_indices_2 = []
    
    for i, res in enumerate(result):
        if ((res[0] not in dataset_2) & (res[1] not in dataset_1)):
            dataset_2.append(res[0])
            dataset_1.append(res[1])
            errors.append(result[-1])  # The error is stored in result[2]
            matching_indices_1.append(res[-3])
            matching_indices_2.append(res[-2])

            
    # Create a list of all possible pairs with their distances
    #pairs = [(i, j, distances[i, j]) for i in range(len(coords_1)) for j in range(len(coords_2))]

    # Sort pairs by distance
    #pairs.sort(key=lambda x: x[2])

    dataset_1 = []
    dataset_2 = []
    errors = []
    matching_indices_1 = []
    matching_indices_2 = []

    matched_1 = set()
    matched_2 = set()

    for i, j, dist in pairs:
        if i not in matched_1 and j not in matched_2:
            dataset_1.append(list(coords_1[i]) + [atom_indices_1[i]])
            dataset_2.append(list(coords_2[j]) + [atom_indices_2[j]])
            errors.append(dist)
            matching_indices_1.append(i)
            matching_indices_2.append(j)
            matched_1.add(i)
            matched_2.add(j)

    # Pad the shorter list if necessary
    if len(dataset_1) < len(input_list_1):
        for i in range(len(input_list_1)):
            if i not in matched_1:
                dataset_1.append(list(coords_1[i]) + [atom_indices_1[i]])
                dataset_2.append([0, 0, -1])  # -1 indicates a padded point
                errors.append(np.inf)
                matching_indices_1.append(i)
                matching_indices_2.append(-1)

    elif len(dataset_2) < len(input_list_2):
        for j in range(len(input_list_2)):
            if j not in matched_2:
                dataset_2.append(list(coords_2[j]) + [atom_indices_2[j]])
                dataset_1.append([0, 0, -1])  # -1 indicates a padded point
                errors.append(np.inf)
                matching_indices_1.append(-1)
                matching_indices_2.append(j)

    return np.array(dataset_1), np.array(dataset_2), np.array(errors), np.array(matching_indices_1), np.array(matching_indices_2)



def hungarian_zero_padded(input_list_1, input_list_2, errors, matching_indices_1, matching_indices_2):
    """ From https://stackoverflow.com/questions/39016821/minimize-total-distance-between-two-sets-of-points-in-python
    """
    C = cdist(input_list_1[:, :2], input_list_2[:, :2])  # Only use x, y coordinates for distance calculation
    row_ind, col_ind = linear_sum_assignment(C)
    
    # Reorder input lists, errors, and matching indices based on the assignment
    output_array_1 = input_list_1[row_ind]
    output_array_2 = input_list_2[col_ind]
    
    aligned_errors = np.array(errors)[col_ind]
    aligned_matching_indices_1 = np.array(matching_indices_1)[row_ind]
    aligned_matching_indices_2 = np.array(matching_indices_2)[col_ind]
    return output_array_1, output_array_2, aligned_errors, aligned_matching_indices_1, aligned_matching_indices_2



def euclidean_distance_nn(input_list_1, input_list_2):
    

    input_list_1_align, input_list_2_align, errors, matching_indices_1, matching_indices_2 = euclidean_distance_uneven(input_list_1, input_list_2)
    input_list_1_align_part_2 = []
    input_list_2_align_part_2 = []
    errors_ = []
    matching_indices_1_ = []
    matching_indices_2_ = []

    if len(input_list_1) < len(input_list_2):
        for i in input_list_2:
            if not any(np.array_equal(i[:2], j[:2]) for j in input_list_2_align):
                input_list_2_align_part_2.append(i)
        
        input_list_1_align_part_2, input_list_2_align_part_2, errors_, matching_indices_1_, matching_indices_2_ = euclidean_distance_uneven(input_list_1, np.array(input_list_2_align_part_2))

    elif len(input_list_1) > len(input_list_2):
        for i in input_list_1:
            if not any(np.array_equal(i[:2], j[:2]) for j in input_list_1_align):
                
                input_list_1_align_part_2.append(i)
        
        input_list_1_align_part_2, input_list_2_align_part_2, errors_, matching_indices_1_, matching_indices_2_ = euclidean_distance_uneven(np.array(input_list_1_align_part_2), input_list_2)

    input_list_1_align = list(input_list_1_align)
    input_list_2_align = list(input_list_2_align)
    input_list_1_align.extend(list(input_list_1_align_part_2))
    input_list_2_align.extend(list(input_list_2_align_part_2))
    errors.extend(errors_)
    matching_indices_1.extend(matching_indices_1_)
    matching_indices_2.extend(matching_indices_2_)

    return np.array(input_list_1_align), np.array(input_list_2_align), errors, matching_indices_1, matching_indices_2


def euclidean_distance_uneven(input_list_1,input_list_2):
    """This function aligns the closest points with each other based on euclidean distance
    and matches the remaining ones with the zero padding"""

    # calculate euclidean distance
    result = []
    for j in input_list_1:
        for i in input_list_2:
            dst = distance.euclidean(i[:2], j[:2])
            result.append([list(j[:2]),list(i[:2]), j[-1], i[-1],dst])
          
    # sort it for the lowest euclidean distance
    result = sorted(result, key=lambda x:x[-1], reverse=False)
    #import IPython; IPython.embed();
    # This aligns the closest points with each other
    # and compares the remaining with the zero padding
    dataset_1 = []
    dataset_2 = []
    matching_indices_1 = []
    matching_indices_2 = []
    errors = []
    #import IPython; IPython.embed();  

    for i, res in enumerate(result):
        if ((res[0] not in dataset_1) & (res[1] not in dataset_2)):
            
            dataset_1.append(res[0])#,res[3], res[2], res[4]])
            dataset_2.append(res[1])#, res[2], res[3], res[4]])
            matching_indices_1.append(res[2])
            matching_indices_2.append(res[3])
            errors.append(res[4])  # The error is stored in result[2]
    #import IPython; IPython.embed();
            
    ### Here I check if any of the combinations was already seen before
    ### Also the matches with the zero padding
    """  for i, res in result:
        # print(i)
        if ((res[0] not in dataset_2) & (res[1] not in dataset_1)):
            dataset_2.append(res[0])
            dataset_1.append(res[1])
            matching_indices_1.append(-1)
            matching_indices_2.append(-1) """           
    return np.array(dataset_1), np.array(dataset_2), errors, matching_indices_1, matching_indices_2

def hungarian_advanced_euc(input_list_1,input_list_2):
    #Hungarian advanced

    input_list_1_euc, input_list_2_euc, errors, matching_indices_1, matching_indices_2 = euclidean_distance_nn(input_list_1,input_list_2)
    #import IPython; IPython.embed();    
    input_list_1_euc_hung,input_list_2_euc_hung, errors, matching_indices_1, matching_indices_2 = hungarian_zero_padded(input_list_1_euc,input_list_2_euc, errors, matching_indices_1, matching_indices_2)
    return np.array(input_list_1_euc_hung),np.array(input_list_2_euc_hung), errors, matching_indices_1, matching_indices_2

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




# Function to test the similarity calculations
def test_similarity_calculations_HSQC(file_path1, file_path2):
    df1, _, canonicalized_smiles1 = ed.run_HSQC_generation(file_path1)
    df2, _, canonicalized_smiles2 = ed.run_HSQC_generation(file_path2)
    
    overall_error, df1_out, df2_out = similarity_calculations_HSQC(df1, df2, mode="hung_dist_nn", similarity_type="euclidean", error="sum", assignment_plot=False)
    
    print(f"Overall Error: {overall_error}")
    print("\nFirst few rows of df1_out:")
    print(df1_out)
    print("\nFirst few rows of df2_out:")
    print(df2_out)
    return df1_out, df2_out, canonicalized_smiles1, canonicalized_smiles2


    
############################################################################################
####################################### COSY ###############################################
############################################################################################
'''
def similarity_calculations_COSY(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
                            similarity_type=["euclidean","cosine_similarity"],  \
                            error=["sum","avg"], \
                            assignment_plot = True):
    """This function calculates the cosine similarity of xy of two spectra provided by dataframes
    by first normalizing it and then choosing one of two modes
    min_sum: takes the minimum sum of x + y as a sorting criteria
    euc_dist: compares every point with each other point of the spectra and matches them to minimize the error"""

    
    h_dim_1 = list(np.array(df_1['F2 (ppm)'].astype(float))/10-0.5)
    c_dim_1 = list(np.array(df_1['F1 (ppm)'].astype(float))/10-0.5)
    h_dim_2 = list(np.array(df_2['F2 (ppm)'].astype(float))/10-0.5)
    c_dim_2 = list(np.array(df_2['F1 (ppm)'].astype(float))/10-0.5)
    
    # Include original indices
    input_list_1 = np.array([h_dim_1, c_dim_1, df_1.atom_index]).transpose()
    input_list_2 = np.array([h_dim_2, c_dim_2, df_2.atom_index]).transpose()
    #import IPython; IPython.embed();
    #sorting by sum of x,y high to low
    ######### Do alignment algorithm instead
    if mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2, errors, matching_indices_1, matching_indices_2 = hungarian_advanced_euc(input_list_1,input_list_2)
        from scipy.spatial import distance
        #errors_ = np.array([distance.euclidean(a, b) for a, b in zip(input_list_1, input_list_2)])
        
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
        return cos_sim, np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*10]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*10]).transpose()
    
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
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out['F2 (ppm)'] = (df_1_out['F2 (ppm)'] + 0.5) * 10
            df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 10

            
            df_2_out['F2 (ppm)'] = (df_2_out['F2 (ppm)'] + 0.5) * 10
            df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 10
            
            if assignment_plot:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, avg_dist)

            #return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()
            return np.array(avg_dist), df_1_out, df_2_out
        
        elif error=="sum":
            similarity_type = similarity_type + "_" + error
            sum_error = np.array(sum_dist)
            #import IPython; IPython.embed();
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out['F2 (ppm)'] = pd.to_numeric(df_1_out['F2 (ppm)'], errors='coerce')
            df_1_out['F1 (ppm)'] = pd.to_numeric(df_1_out['F1 (ppm)'], errors='coerce')

            df_1_out['F2 (ppm)'] = (df_1_out['F2 (ppm)'] + 0.5) * 10
            df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 10

            df_2_out['F2 (ppm)'] = pd.to_numeric(df_2_out['F2 (ppm)'], errors='coerce')
            df_2_out['F1 (ppm)'] = pd.to_numeric(df_2_out['F1 (ppm)'], errors='coerce')            
            df_2_out['F2 (ppm)'] = (df_2_out['F2 (ppm)'] + 0.5) * 10
            df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 10
            
            if assignment_plot == True:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, float(sum_error))
                pass
           
            return sum_error,  df_1_out, df_2_out

# Function to test the similarity calculations
def test_similarity_calculations_COSY(file_path1, file_path2):
    df1, stereo_smi1 = ed.run_COSY_generation(file_path1)
    df1['atom_index'] = df1['atom_index_1'].astype(str) + '_' + df1['atom_index_2'].astype(str)
    df1 = df1[(df1['F2 (ppm)'] >= df1['F1 (ppm)'])]
    
    df2, stereo_smi2 = ed.run_COSY_generation(file_path2)
    df2['atom_index'] = df2['atom_index_1'].astype(str) + '_' + df2['atom_index_2'].astype(str)
    df2 = df2[(df2['F2 (ppm)'] >= df2['F1 (ppm)'])]

    #import IPython; IPython.embed();
    overall_error, df1_out, df2_out = similarity_calculations_COSY(df1, df2, mode="hung_dist_nn", similarity_type="euclidean", error="sum", assignment_plot=False)
    
    print(f"Overall Error: {overall_error}")
    print("\nFirst few rows of df1_out:")
    print(df1_out)
    print("\nFirst few rows of df2_out:")
    print(df2_out)
    return df1_out, df2_out, stereo_smi1, stereo_smi2'''



def similarity_calculations_COSY(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
                            similarity_type=["euclidean","cosine_similarity"],  \
                            error=["sum","avg"], \
                            assignment_plot = True):
    """This function calculates the cosine similarity of xy of two spectra provided by dataframes
    by first normalizing it and then choosing one of two modes
    min_sum: takes the minimum sum of x + y as a sorting criteria
    euc_dist: compares every point with each other point of the spectra and matches them to minimize the error"""

    
    h_dim_1 = list(np.array(df_1['F2 (ppm)'].astype(float))/10-0.5)
    c_dim_1 = list(np.array(df_1['F1 (ppm)'].astype(float))/10-0.5)
    h_dim_2 = list(np.array(df_2['F2 (ppm)'].astype(float))/10-0.5)
    c_dim_2 = list(np.array(df_2['F1 (ppm)'].astype(float))/10-0.5)
    
    # Include original indices
    input_list_1 = np.array([h_dim_1, c_dim_1, df_1.atom_index]).transpose()
    input_list_2 = np.array([h_dim_2, c_dim_2, df_2.atom_index]).transpose()
    #import IPython; IPython.embed();
    #sorting by sum of x,y high to low
    ######### Do alignment algorithm instead
    if mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2, errors, matching_indices_1, matching_indices_2 = hungarian_advanced_euc(input_list_1,input_list_2)
        from scipy.spatial import distance
        #errors_ = np.array([distance.euclidean(a, b) for a, b in zip(input_list_1, input_list_2)])
        #print("in COSY sim func")
        #import IPython; IPython.embed();

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
        return cos_sim, np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*10]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*10]).transpose()
    
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
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out['F2 (ppm)'] = (df_1_out['F2 (ppm)'] + 0.5) * 10
            df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 10

            
            df_2_out['F2 (ppm)'] = (df_2_out['F2 (ppm)'] + 0.5) * 10
            df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 10
            
            if assignment_plot:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, avg_dist)

            #return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()
            return np.array(avg_dist), df_1_out, df_2_out
        
        elif error=="sum":
            similarity_type = similarity_type + "_" + error
            sum_error = np.array(sum_dist)
            #import IPython; IPython.embed();
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['F2 (ppm)', 'F1 (ppm)', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out['F2 (ppm)'] = pd.to_numeric(df_1_out['F2 (ppm)'], errors='coerce')
            df_1_out['F1 (ppm)'] = pd.to_numeric(df_1_out['F1 (ppm)'], errors='coerce')

            df_1_out['F2 (ppm)'] = (df_1_out['F2 (ppm)'] + 0.5) * 10
            df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 10

            df_2_out['F2 (ppm)'] = pd.to_numeric(df_2_out['F2 (ppm)'], errors='coerce')
            df_2_out['F1 (ppm)'] = pd.to_numeric(df_2_out['F1 (ppm)'], errors='coerce')            
            df_2_out['F2 (ppm)'] = (df_2_out['F2 (ppm)'] + 0.5) * 10
            df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 10
            
            if assignment_plot == True:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, float(sum_error))
                pass
           
            return sum_error,  df_1_out, df_2_out

# Function to test the similarity calculations
def test_similarity_calculations_COSY(file_path1, file_path2):
    df1, stereo_smi1 = ed.run_COSY_generation(file_path1)
    df1['atom_index'] = df1['atom_index_1'].astype(str) + '_' + df1['atom_index_2'].astype(str)
    df1 = df1[(df1['F2 (ppm)'] >= df1['F1 (ppm)'])]
    
    df2, stereo_smi2 = ed.run_COSY_generation(file_path2)
    df2['atom_index'] = df2['atom_index_1'].astype(str) + '_' + df2['atom_index_2'].astype(str)
    df2 = df2[(df2['F2 (ppm)'] >= df2['F1 (ppm)'])]

    #import IPython; IPython.embed();
    overall_error, df1_out, df2_out = similarity_calculations_COSY(df1, df2, mode="hung_dist_nn", similarity_type="euclidean", error="sum", assignment_plot=False)
    
    print(f"Overall Error: {overall_error}")
    print("\nFirst few rows of df1_out:")
    print(df1_out)
    print("\nFirst few rows of df2_out:")
    print(df2_out)
    return df1_out, df2_out, stereo_smi1, stereo_smi2




    
############################################################################################
####################################### 13C ###############################################
############################################################################################
    
def similarity_calculations_13C(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
                            similarity_type=["euclidean","cosine_similarity"],  \
                            error=["sum","avg"], \
                            assignment_plot = True):
    """This function calculates the cosine similarity of xy of two spectra provided by dataframes
    by first normalizing it and then choosing one of two modes
    min_sum: takes the minimum sum of x + y as a sorting criteria
    euc_dist: compares every point with each other point of the spectra and matches them to minimize the error"""
  
    h_dim_1 = list(np.array(list(df_1["shifts"].astype(float)))/200-0.5)
    c_dim_1 = [1 for i in range(len(df_1["shifts"]))]
    h_dim_2 = list(np.array(df_2["shifts"].astype(float))/200-0.5)
    c_dim_2 = [1 for i in range(len(df_2["shifts"]))]
    
    # Include original indices
    input_list_1 = np.array([h_dim_1, c_dim_1, df_1.atom_index]).transpose()
    input_list_2 = np.array([h_dim_2, c_dim_2, df_2.atom_index]).transpose()
    #import IPython; IPython.embed();
    #sorting by sum of x,y high to low
    ######### Do alignment algorithm instead
    if mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2, errors, matching_indices_1, matching_indices_2 = hungarian_advanced_euc(input_list_1,input_list_2)
        from scipy.spatial import distance
        #errors_ = np.array([distance.euclidean(a, b) for a, b in zip(input_list_1, input_list_2)])
        #print("in COSY sim func")
        #import IPython; IPython.embed();

    # if similarity_type == "cosine_similarity":
    #     from scipy.spatial import distance
    #     list_points_1 = np.array(input_list_1, dtype=object)
    #     list_points_2 = np.array(input_list_2, dtype=object)
    #     Aflat = np.hstack(list_points_1)
    #     Bflat = np.hstack(list_points_2)
    #     # Aflat = Aflat - Aflat.mean()
    #     # Bflat = Bflat - Bflat.mean()
    #     cos_sim = 1 - distance.cosine(Aflat, Bflat)
    #     if assignment_plot == True:
    #         plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, cos_sim)
    #         pass
    #     return cos_sim, np.array([(input_list_1[:,0]+0.5)*200,(input_list_1[:,1])]).transpose(), np.array([(input_list_2[:,0]+0.5)*200,(input_list_2[:,1]]).transpose()
    
    if similarity_type == "euclidean":
        
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
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['shifts', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['shifts', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out['shifts'] = (df_1_out['shifts'] + 0.5) * 200
            #df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 10
            
            df_2_out['shifts'] = (df_2_out['shifts'] + 0.5) * 200
            #df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 10
            
            if assignment_plot:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, avg_dist)

            #return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()
            return np.array(avg_dist), df_1_out, df_2_out
        
        elif error=="sum":
            similarity_type = similarity_type + "_" + error
            sum_error = np.array(sum_dist)
            #import IPython; IPython.embed();
            
           # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['shifts', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['shifts', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out["shifts"] = pd.to_numeric(df_1_out['shifts'], errors='coerce')
            #df_1_out['F1 (ppm)'] = pd.to_numeric(df_1_out['F1 (ppm)'], errors='coerce')

            df_1_out['shifts'] = (df_1_out['shifts'] + 0.5) * 200
            #df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 10

            df_2_out['shifts'] = pd.to_numeric(df_2_out['shifts'], errors='coerce')
            #df_2_out['F1 (ppm)'] = pd.to_numeric(df_2_out['F1 (ppm)'], errors='coerce')            
            df_2_out['shifts'] = (df_2_out['shifts'] + 0.5) * 200
            #df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 10
            
            if assignment_plot == True:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, float(sum_error))
                pass
           
            return sum_error, df_1_out, df_2_out



############################################################################################
######################################## 1H ###############################################
############################################################################################
    
def similarity_calculations_1H(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
                            similarity_type=["euclidean","cosine_similarity"],  \
                            error=["sum","avg"], \
                            assignment_plot = True):
    """This function calculates the cosine similarity of xy of two spectra provided by dataframes
    by first normalizing it and then choosing one of two modes
    min_sum: takes the minimum sum of x + y as a sorting criteria
    euc_dist: compares every point with each other point of the spectra and matches them to minimize the error"""

    # print("similarity_calculations_1H")
    # import IPython; IPython.embed();
    h_dim_1 = list(np.array(list(df_1["shifts_orig"].astype(float)))/10-0.5)
    c_dim_1 = [1 for i in range(len(df_1["shifts_orig"]))]
    h_dim_2 = list(np.array(list(df_2["shifts_orig"].astype(float)))/10-0.5)
    c_dim_2 = [1 for i in range(len(df_2["shifts_orig"]))]

    # Include original indices
    input_list_1 = np.array([h_dim_1, c_dim_1, df_1.atom_index]).transpose()
    input_list_2 = np.array([h_dim_2, c_dim_2, df_2.atom_index]).transpose()
    #import IPython; IPython.embed();
    #sorting by sum of x,y high to low
    ######### Do alignment algorithm instead
    if mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2, errors, matching_indices_1, matching_indices_2 = hungarian_advanced_euc(input_list_1,input_list_2)
        from scipy.spatial import distance
        #errors_ = np.array([distance.euclidean(a, b) for a, b in zip(input_list_1, input_list_2)])
        #print("in COSY sim func")
        #import IPython; IPython.embed();

    # if similarity_type == "cosine_similarity":
    #     from scipy.spatial import distance
    #     list_points_1 = np.array(input_list_1, dtype=object)
    #     list_points_2 = np.array(input_list_2, dtype=object)
    #     Aflat = np.hstack(list_points_1)
    #     Bflat = np.hstack(list_points_2)
    #     # Aflat = Aflat - Aflat.mean()
    #     # Bflat = Bflat - Bflat.mean()
    #     cos_sim = 1 - distance.cosine(Aflat, Bflat)
    #     if assignment_plot == True:
    #         plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, cos_sim)
    #         pass
    #     return cos_sim, np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*10]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*10]).transpose()
    
    if similarity_type == "euclidean":
        
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
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['shifts_orig', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['shifts_orig', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out['shifts_orig'] = (df_1_out['shifts_orig'] + 0.5) * 10
            # df_1_out['intensity'] = (df_1_out['intensity'] + 0.5) * 10

            
            df_2_out['shifts_orig'] = (df_2_out['shifts_orig'] + 0.5) * 10
            # df_2_out['intensity'] = (df_2_out['intensity'] + 0.5) * 10
            
            if assignment_plot:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, avg_dist)

            #return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*10,(input_list_1[:,1]+0.5)*200]).transpose(), np.array([(input_list_2[:,0]+0.5)*10,(input_list_2[:,1]+0.5)*200]).transpose()
            return np.array(avg_dist), df_1_out, df_2_out
        
        elif error=="sum":
            similarity_type = similarity_type + "_" + error
            sum_error = np.array(sum_dist)
            #import IPython; IPython.embed();
            
            # Prepare output dataframes with error and matching index information
            df_1_out = pd.DataFrame(np.column_stack([input_list_1, errors, matching_indices_1, matching_indices_2]), 
                                    columns=['shifts_orig', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            df_2_out = pd.DataFrame(np.column_stack([input_list_2, errors, matching_indices_2, matching_indices_1]), 
                                    columns=['shifts_orig', 'intensity', 'Error', 'Self_Index', 'Matching_Index'])
            # Convert back to original scale
            df_1_out['shifts_orig'] = pd.to_numeric(df_1_out['shifts_orig'], errors='coerce')
            # df_1_out['F1 (ppm)'] = pd.to_numeric(df_1_out['F1 (ppm)'], errors='coerce')

            df_1_out['shifts_orig'] = (df_1_out['shifts_orig'] + 0.5) * 10
            #df_1_out['F1 (ppm)'] = (df_1_out['F1 (ppm)'] + 0.5) * 10

            df_2_out['shifts_orig'] = pd.to_numeric(df_2_out['shifts_orig'], errors='coerce')
            # df_2_out['F1 (ppm)'] = pd.to_numeric(df_2_out['F1 (ppm)'], errors='coerce')            
            df_2_out['shifts_orig'] = (df_2_out['shifts_orig'] + 0.5) * 10
            #df_2_out['F1 (ppm)'] = (df_2_out['F1 (ppm)'] + 0.5) * 10
            
            if assignment_plot == True:
                plot_assignment_points(input_list_1, input_list_2, mode, similarity_type, float(sum_error))
                pass
           
            return sum_error,  df_1_out, df_2_out            