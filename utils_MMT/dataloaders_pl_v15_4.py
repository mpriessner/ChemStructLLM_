##### before modifying collate_fn function
# Standard library imports
import ast
import pickle
import random
import re
import threading
import os
from ast import literal_eval

# Third-party imports
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
tqdm.pandas()  # Enables progress_apply for pandas

# Local application/library specific imports
import utils_MMT.helper_functions_pl_v15_4 as hf
#import utils.nmr_calculation_from_dft_v15_4
#import utils.train_test_functions_pl_v15_4 as ttf
from utils_MMT.smi_augmenter_v15_4 import SMILESAugmenter
from torch.nn.utils.rnn import pad_sequence


# SmilesEnumerator class for SMILES generation, vectorization and devectorization
class SmilesEnumerator(object):
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical


    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        
    def fit(self, smiles, extra_chars=[], extra_pad = 5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset
        
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None # Invalid SMILES
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        
        if self.leftpad:
            for i,ss in enumerate(smiles):
                if self.enumerate: 
                    ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j,c in enumerate(ss):
                    one_hot[i,j+diff,self._char_to_int[c]] = 1
            return one_hot
        else:
            for i,ss in enumerate(smiles):
                if self.enumerate: 
                    ss = self.randomize_smiles(ss)
                for j,c in enumerate(ss):
                    one_hot[i,j,self._char_to_int[c]] = 1
            return one_hot

      
    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """       
        smiles = []
        for v in vect:
            #mask v 
            v=v[v.sum(axis=1)==1]
            #Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)  


def pollute_HSQC_data(HSQC_df, noise_peaks, noise_num_list):
    """
    Add noise peaks to a given HSQC dataframe based on the given noise peaks and peak counts.
    Args:
    acd_df: pandas dataframe of HSQC data with columns "F2 (ppm)", "F1 (ppm)", and "direction".
    noise_peaks: list of lists with the noise peak coordinates in the format [[x1, y1, int1], [x2, y2, int2], ...].
    noise_num_list: list of integers representing the number of peaks in each spectrum to use as a reference for noise peak addition.

    Returns:
    HSQC_df: updated pandas dataframe with added noise peaks.
    """
    # Select random peak number
    num_peaks = random.choice(noise_num_list)

    # Select random noise peaks
    selected_noise = random.sample(noise_peaks, num_peaks)

    # Add noise peaks to simulated data
    for peak in selected_noise:
        x, y, _ = peak
        intensity = 1 if np.random.random() < 0.5 else -1
        HSQC_df = HSQC_df.append({"F2 (ppm)": x, "F1 (ppm)": y, 'direction': intensity}, ignore_index=True)

    return HSQC_df

# DatasetSpectrumTransformer class for handling spectrum data
class MultimodalData(Dataset):
    def __init__(self, config, stoi, stoi_MF, mode):
        self.config = config
        self.stoi = stoi
        self.stoi_MF = stoi_MF
        self.mode = mode
        self.smiles_randomizer = SMILESAugmenter(restricted=True)
        self.fingerprint_size = config.fingerprint_size
        # Create a dictionary to easily access datasets
        self.possible_datasets = ["1H", "13C", "HSQC", "COSY", "IR"]

        if self.config.data_type == "sgnn":
            self.ref_data = self._load_sgnn_data()

 
    def _calculate_min_max(self, ref_data):
        
        if self.config.data_type=="sgnn":
            ref_data['MW'] = ref_data['SMILES'].progress_apply(hf.calculate_mol_weight)

            # Find the minimum and maximum molecular weights
            self.config.train_weight_min  = ref_data['MW'].min()
            self.config.train_weight_max = ref_data['MW'].max()
            return ref_data

    def _save_pkl_dict(self, config, original_dict):
        print("Proecessing data to save as pkl file")
        reshuffled_dict = {}

        # Defining the keys that should be reshuffled
        keys_to_reshuffle = ['1H', '13C', 'HSQC', 'COSY', 'IR', 'SMILES']

        # Iterate through the keys of the original dictionary
        for key in original_dict.keys():
            # Iterate through the sub-dictionary
            for sample_id, data in original_dict[key].iterrows():
                # Check if the sample_id is already in the reshuffled dictionary
                if sample_id not in reshuffled_dict:

                    # If not, initialize a new entry with default values (empty lists for missing data)
                    reshuffled_dict[sample_id] = {key: [] for key in keys_to_reshuffle}
                    reshuffled_dict[sample_id]['SMILES'] = data.SMILES  # Initialize SMILES with None

                # Update the corresponding key-value pair based on the original dictionary
                if key in ['1H', '13C', 'HSQC', 'COSY', 'IR']:
                    reshuffled_dict[sample_id][key] = data.drop('SMILES').values.tolist()  # Excluding the SMILES column
        
        # Step 1: Split the path into directory and filename
        dir_path, filename = os.path.split(config.csv_1H_path_SGNN)

        # Step 2: Generate a random number
        random_number = random.randint(1, 1000000)

        # Step 3: Insert the random number before the .pickle extension
        filename_base, file_extension = os.path.splitext(filename)
        new_filename = f"{filename_base}_{random_number}{'.pkl'}"

        # Step 4: Reconstruct the full path
        new_file_path = os.path.join(dir_path, new_filename)
        # Save reshuffled_dict to a pickle file
        config.pickle_file_path = new_file_path
        print(config.pickle_file_path)
        with open(config.pickle_file_path, 'wb') as f:
            pickle.dump(reshuffled_dict, f)
            print("Pickle saved.")
        return reshuffled_dict

    def _load_sgnn_data(self):

        # Corresponding config paths
        config_paths = {
            "1H": self.config.csv_1H_path_SGNN,
            "13C": self.config.csv_13C_path_SGNN,
            "HSQC": self.config.csv_HSQC_path_SGNN,
            "COSY": self.config.csv_COSY_path_SGNN,
            "IR": self.config.csv_IR_MF_path
            }
               
        if os.path.exists(self.config.pickle_file_path):# and self.mode != "val":
            # Load data from pickle file if it exists
            with open(self.config.pickle_file_path, 'rb') as f:
                self.data_dict = pickle.load(f)
                print("Pickle Loaded.")
        else:
            self.data_dict = {}
            # Checking if pickle file exists for faster loading, else load from CSV and save as Pickle
           
            for dataset in self.possible_datasets:
                if dataset in self.config.training_mode:
                    config_path = config_paths.get(dataset)
                    try:
                        # Read the CSV file
                        df = pd.read_csv(config_path)
                       
                        # Check if 'Unnamed: 0' column exists and drop it if it does
                        if 'Unnamed: 0' in df.columns:
                            df = df.drop('Unnamed: 0', axis=1)

                        df.set_index('sample-id', inplace=True)
                        self.data_dict[dataset] = df
                    except FileNotFoundError:
                        print(f"Warning: File for dataset {dataset} not found. Skipping...")
                else:
                    print(f"Warning: Dataset {dataset} not selected.")
            self.data_dict = self._save_pkl_dict(self.config, self.data_dict)
        
        # Add handling for validation data
        if self.mode == "val":
            try:
                self.ref_data = pd.read_csv(self.config.csv_path_val)
                self.ref_data = self.ref_data.iloc[:self.config.data_size]
                #self.ref_data.set_index('sample-id', inplace=True)
            except FileNotFoundError:
                print("Warning: Validation file not found.")
        else:
            self.ref_data = pd.read_csv(config_paths[self.config.ref_data_type])
            self.ref_data = self.ref_data.iloc[:self.config.data_size]
            self._calculate_min_max(self.ref_data)
            if self.mode != "all": 
                self.ref_data = self._load_and_split_data(self.ref_data)
            #self.ref_data = self.ref_data.reset_index(drop=True)
        return self.ref_data
    

    def _zero_pad(self, data, pad_length, dimensions=1):
        """Zero pad array data to pad_length and return a mask."""
        masks_list = []
        padded_spectra_list = []

        if dimensions == 1:
            mask = torch.ones(pad_length).long()
            data_tensor = torch.tensor(data)
            if len(data) >= pad_length:
                padded_spectra_list.append(data_tensor[:pad_length])
                masks_list.append(mask)
            else:
                mask[:len(data)] = 0
                padded_data = torch.cat((data_tensor, torch.zeros(pad_length - len(data))), dim=0)
                padded_spectra_list.append(padded_data)
                masks_list.append(mask)

        elif dimensions == 2:
            mask = torch.ones(pad_length).long()
            padded_data = [list(item) for item in data]
            
            if len(padded_data) >= pad_length:
                mask[:len(padded_data)] = 0
                padded_spectra_list.append(torch.tensor(padded_data[:pad_length]))
                masks_list.append(mask)
            else:
                mask[:len(padded_data)] = 0
                while len(padded_data) < pad_length:
                    padded_data.append([0, 0])
                padded_spectra_list.append(torch.tensor(padded_data[:pad_length]))
                masks_list.append(mask)

        return torch.stack(padded_spectra_list).squeeze(0), torch.stack(masks_list).squeeze(0)

    def _load_and_split_data(self, data):
        
        # Split the data into train and test sets

        train_data = data.iloc[:int(self.config.tr_te_split * len(data))]
        test_data = data.iloc[int((self.config.tr_te_split * len(data))):]
        # Reset the indices
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        # Select Train or test data
        data = train_data if self.mode=="train" else test_data    
        return data
    
    # Function to search for a row with a given entry in the 'sample-id' column and return the 'shifts' value
    def _find_shifts_by_sample_id(self, df, idx):
        #row = df[df['sample-id'] == sample_id]
        try:
            shifts_value = df.at[idx, 'shifts']
            return ast.literal_eval(shifts_value)
        except KeyError:
            return None

    
    def _load_IR_data(self, config, sample_id):
        file_path = f"{config.IR_data_folder}/{sample_id}.csv"
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Get the 'spectra' column and convert it to a list
        spectra_list = df['spectra'].tolist()
        max_val = max(spectra_list)
        
        # Initialize mask list
        mask = np.zeros(config.input_dim_IR, dtype=int)

        # Reduce to a given spectrum dimension
        average_span = len(spectra_list) / config.input_dim_IR
        binned_spectrum = np.zeros(config.input_dim_IR)
        start = 0
        for i in range(config.input_dim_IR):
            end = start + average_span  # round to the nearest integer
            binned_spectrum[i] = np.mean(spectra_list[round(start):round(end)])/max_val
            start = end  # update the start for the next iteration
            
        return torch.tensor(binned_spectrum), torch.tensor(mask)

    def _normalize_shifts_2D_spectra(self, shifts_2D, mode):
        # Initialize an empty list to store the normalized shifts
        normalized_shifts = []
        
        if mode == "HSQC":
            # Loop through each sublist in shifts_HSQC
            for shift_pair in shifts_2D:
                # Normalize the first value by 10 and the second value by 200
                normalized_pair = [shift_pair[0] / 10, shift_pair[1] / 200]
                # Append the normalized pair to the list
                normalized_shifts.append(normalized_pair)
        elif mode == "COSY":
            # Loop through each sublist in shifts_HSQC
            for shift_pair in shifts_2D:
                # Normalize the first value by 10 and the second value by 10
                normalized_pair = [shift_pair[0] / 10, shift_pair[1] / 10]
                # Append the normalized pair to the list
                normalized_shifts.append(normalized_pair)    
        return normalized_shifts

    
    def _create_empty_data_and_mask(self, padding_points_number, dimensions=1):
        """
        Create empty data and a corresponding mask filled with ones.

        Parameters:
        - padding_points_number: The length of the tensor.
        - dimensions: The dimensions of the tensor (1 for 1D, 2 for 2D, 3 for 3D)

        Returns:
        - empty_data: A tensor filled with zeros.
        - mask: A tensor filled with ones.
        """
        if dimensions == 1:
            empty_data = torch.zeros(padding_points_number)
            mask = torch.ones(padding_points_number).long()

        elif dimensions == 2:
            empty_data = torch.zeros(padding_points_number, 2)
            mask = torch.ones(padding_points_number).long()

        elif dimensions == 3:
            empty_data = torch.zeros(padding_points_number, 3)  
            mask = torch.ones(padding_points_number).long()
        return empty_data, mask

    
    def _encode_smiles(self, stoi, c_smi):
        """
        Encodes the tokenized SMILES string into integers.

        Parameters:
        - stoi: Dictionary mapping tokens to integers.
        - c_smi: The canonical SMILES string.
        - two_letter_atoms: A list of two-letter atoms (e.g., ['Cl', 'Br']).

        Returns:
        - encoded: A list of integers representing the tokenized SMILES string.
        """
        tokens = hf.tokenize_smiles(c_smi, hf.two_char_symbols)
        return [stoi.get(token, stoi.get('<UNK>')) for token in tokens]


    def _encode_MF(self, stoi_MF, MF):
         # Use a regular expression to capture element symbols and their counts
        # Elements start with an uppercase letter followed by zero or more lowercase letters.
        # Counts are one or more digits.
        pattern = r'([A-Z][a-z]*)(\d{1,2})?'
        tokens = []
        for element, count in re.findall(pattern, MF):
            tokens.append(element)
            if count:  # Only add the count if it's present (i.e., not None and not an empty string)
                tokens.append(count)
        return [stoi_MF.get(token, stoi_MF.get('<UNK>')) for token in tokens]

    def __len__(self):
        return len(self.ref_data)
    
    def __getitem__(self, idx):
        item_dict = {}
        #import IPython; IPython.embed();
        if  self.config.data_type in ["sgnn"]:
            
            ### Load SMILES and Molecular Formula 
            sample_id = self.ref_data.iloc[idx]["sample-id"]
            #import IPython; IPython.embed();

            data_entry = self.data_dict[sample_id]
            smi = data_entry['SMILES']
            #smi = self.ref_data.at[sample_id, 'SMILES']  
            mol = Chem.MolFromSmiles(smi)
            MF = rdMolDescriptors.CalcMolFormula(mol)

            ### Canonicalize SMILES
            c_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=self.config.isomericSmiles)     
            trg_MW = torch.tensor(rdMolDescriptors.CalcExactMolWt(mol))
            item_dict['trg_MW'] = trg_MW

            ### Randomize if selected SMILES
            if self.config.smi_randomizer:
                # Randomize smiles
                c_smi = self.smiles_randomizer([c_smi])[0]

            actual_spectra = [s for s in self.possible_datasets if s in self.config.training_mode]
            #### Load 1H Spectrum
            #import IPython; IPython.embed();
            if '1H' in actual_spectra:
                try:
                    shifts_1H = data_entry["1H"][0]
                    shifts_1H = ast.literal_eval(shifts_1H)
                    #shifts_1H = self._find_shifts_by_sample_id(self.data_dict["1H"], sample_id)
                    shifts_1H = [[shift[0] / 10.0, shift[1]] for shift in shifts_1H]
                    src_1H, mask_1H = self._zero_pad(shifts_1H, self.config.padding_points_number, dimensions=self.config.input_dim_1H)    
                    #print("Time for loading 3 and Molecular Formula:", time.time() - start_time)
                    #start_time = time.time()  
                    chance = self.config.blank_percentage
                except:
                    chance = 1
                    #import IPython; IPython.embed();
                    #print(sample_id)
                if random.random() < chance:  # x% chance to blank out the data
                    src_1H, mask_1H = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_1H)
                    actual_spectra.remove('1H')
                item_dict['src_1H'] = src_1H
                item_dict['mask_1H'] = mask_1H
            else:
                src_1H, mask_1H = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_1H)
                item_dict['src_1H'] = src_1H
                item_dict['mask_1H'] = mask_1H

            #### Load 13C Spectrum
            if '13C' in actual_spectra:
                try:
                    shifts_13C = data_entry['13C'][0]
                    shifts_13C = ast.literal_eval(shifts_13C)
                    #shifts_13C = self._find_shifts_by_sample_id(self.data_dict["13C"], sample_id)
                    shifts_13C = [shift / 200.0 for shift in shifts_13C]
                    src_13C, mask_13C = self._zero_pad(shifts_13C, self.config.padding_points_number, dimensions=self.config.input_dim_13C)
                    chance = self.config.blank_percentage
                except:
                    chance = 1
                    #import IPython; IPython.embed();
                    #print(sample_id)
                if random.random() < chance:  # x% chance to blank out the data
                    src_13C, mask_13C = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_13C)
                    actual_spectra.remove('13C')
                item_dict['src_13C'] = src_13C
                item_dict['mask_13C'] = mask_13C   
            else: 
                src_13C, mask_13C = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_13C)
                item_dict['src_13C'] = src_13C
                item_dict['mask_13C'] = mask_13C  

            #### Load HSQC Spectrum

            if 'HSQC' in actual_spectra:
                try:
                    shifts_HSQC = data_entry['HSQC'][0]
                    shifts_HSQC = ast.literal_eval(shifts_HSQC)
                    # shifts_HSQC = self._find_shifts_by_sample_id(self.data_dict["HSQC"], sample_id)
                    shifts_HSQC = self._normalize_shifts_2D_spectra(shifts_HSQC, "HSQC")
                    src_HSQC, mask_HSQC = self._zero_pad(shifts_HSQC, self.config.padding_points_number, dimensions=self.config.input_dim_HSQC)
                    chance = self.config.blank_percentage
                except:
                    chance = 1
                    #import IPython; IPython.embed();
                    #print(sample_id)
                if random.random() < chance:  # x% chance to blank out the data
                    src_HSQC, mask_HSQC = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_HSQC)
                    # Create the tensor
                    actual_spectra.remove('HSQC')
                item_dict['src_HSQC'] = src_HSQC
                item_dict['src_HSQC_'] = src_HSQC      ### For Error analysis ablation study
                item_dict['mask_HSQC'] = mask_HSQC  
                item_dict['mask_HSQC_'] = mask_HSQC   

                #item_dict['tensor_HSQC'] = tensor_HSQC
            else:
                src_HSQC, mask_HSQC = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_HSQC)
                try:
                    shifts_HSQC_ = data_entry['HSQC'][0]
                    shifts_HSQC_ = ast.literal_eval(shifts_HSQC_)
                    # shifts_HSQC = self._find_shifts_by_sample_id(self.data_dict["HSQC"], sample_id)
                    shifts_HSQC_ = self._normalize_shifts_2D_spectra(shifts_HSQC_, "HSQC")
                    src_HSQC_, mask_HSQC_ = self._zero_pad(shifts_HSQC_, self.config.padding_points_number, dimensions=self.config.input_dim_HSQC)
                except:
                    src_HSQC_ = src_HSQC
                    mask_HSQC_ = mask_HSQC
                    pass
                # Create the tensor
                item_dict['src_HSQC'] = src_HSQC
                item_dict['src_HSQC_'] = src_HSQC_       ### For Error analysis ablation study
                item_dict['mask_HSQC'] = mask_HSQC   
                item_dict['mask_HSQC_'] = mask_HSQC_   
                
            #### Load COSY Spectrum
            if 'COSY' in actual_spectra:
                try:
                    shifts_COSY = data_entry['COSY'][0]
                    shifts_COSY = ast.literal_eval(shifts_COSY)
                    #shifts_COSY = self._find_shifts_by_sample_id(self.data_dict["COSY"], sample_id)
                    shifts_COSY = self._normalize_shifts_2D_spectra(shifts_COSY, "COSY")
                    src_COSY, mask_COSY = self._zero_pad(shifts_COSY, self.config.padding_points_number, dimensions=self.config.input_dim_COSY)
                    chance = self.config.blank_percentage
                except:
                    chance = 1
                    #print(sample_id)
                if random.random() < chance:  # x% chance to blank out the data
                    src_COSY, mask_COSY = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_HSQC)
                    actual_spectra.remove('COSY')
                item_dict['src_COSY'] = src_COSY               
                item_dict['src_COSY_'] = src_COSY      ### For Error analysis ablation study         
                item_dict['mask_COSY'] = mask_COSY
                item_dict['mask_COSY_'] = mask_COSY
            else:
                src_COSY, mask_COSY = self._create_empty_data_and_mask(self.config.padding_points_number, dimensions=self.config.input_dim_HSQC)
                try:
                    shifts_COSY_ = data_entry['COSY'][0]
                    shifts_COSY_ = ast.literal_eval(shifts_COSY_)
                    #shifts_COSY = self._find_shifts_by_sample_id(self.data_dict["COSY"], sample_id)
                    shifts_COSY_ = self._normalize_shifts_2D_spectra(shifts_COSY_, "COSY")
                    src_COSY_, mask_COSY_ = self._zero_pad(shifts_COSY_, self.config.padding_points_number, dimensions=self.config.input_dim_COSY)
                except:
                    src_COSY_ = src_COSY
                    mask_COSY_ = mask_COSY
                    pass
                item_dict['src_COSY'] = src_COSY        
                item_dict['src_COSY_'] = src_COSY_      ### For Error analysis ablation study         
                item_dict['mask_COSY'] = mask_COSY
                item_dict['mask_COSY_'] = mask_COSY_
            
            #### Load IR Spectrum
            if 'IR' in actual_spectra:
                try:
                    src_IR, mask_IR = self._load_IR_data(self.config, sample_id)
                    chance = self.config.blank_percentage
                except:
                    chance = 1
                    #print(sample_id)
                if random.random() < chance:  # x% chance to blank out the data
                    src_IR, mask_IR = self._create_empty_data_and_mask(self.config.input_dim_IR, dimensions=1)
                    actual_spectra.remove('IR')
                item_dict['src_IR'] = src_IR
                item_dict['mask_IR'] = mask_IR    
            else:
                src_IR, mask_IR = self._create_empty_data_and_mask(self.config.input_dim_IR, dimensions=1)
                item_dict['src_IR'] = src_IR
                item_dict['mask_IR'] = mask_IR        

            # in that case load at least 1H and 13C again
            if actual_spectra == []:
                #### Load 1H Spectrum
                try:
                    #### Load 1H Spectrum
                    shifts_1H = data_entry['1H'][0]
                    shifts_1H = ast.literal_eval(shifts_1H)
                    # shifts_1H = self._find_shifts_by_sample_id(self.data_dict["1H"], sample_id)
                    shifts_1H = [[shift[0] / 10.0, shift[1]] for shift in shifts_1H]
                    src_1H, mask_1H = self._zero_pad(shifts_1H, self.config.padding_points_number, dimensions=self.config.input_dim_1H)
                    item_dict['src_1H'] = src_1H
                    item_dict['mask_1H'] = mask_1H       
                except:
                # print("extra H failed")
                    #print(sample_id)
                    pass                  
                try:
                    #### Load 13C Spectrum
                    shifts_13C = data_entry['13C'][0]
                    shifts_13C = ast.literal_eval(shifts_13C)
                    # shifts_13C = self._find_shifts_by_sample_id(self.data_dict["13C"], sample_id)
                    shifts_13C = [shift / 200.0 for shift in shifts_13C]
                    src_13C, mask_13C = self._zero_pad(shifts_13C, self.config.padding_points_number, dimensions=self.config.input_dim_13C) 
                    item_dict['src_13C'] = src_13C
                    item_dict['mask_13C'] = mask_13C                             
                except:
                    #print("extra C failed")
                    #print(sample_id)
                    #import IPython; IPython.embed();
                    pass  

            #import IPython; IPython.embed();
            #print(c_smi)
            ### Load target SMI == src_MS
            trg_enc_SMI = self._encode_smiles(self.stoi, c_smi)
            trg_enc_SMI = [self.stoi["<SOS>"]] + trg_enc_SMI + [self.stoi["<EOS>"]] 
            padd = [0 for i in range(self.config.padding_points_number-len(trg_enc_SMI))]
            val = [0 for i in range(len(trg_enc_SMI))]
            src_MS = torch.tensor(trg_enc_SMI + padd)   ### trg_enc_SMI = is also the encoding for src_MS
            inv_pad = list(map(lambda x: 1 if x == 0 else x, padd))
            mask_MS = torch.tensor(val + inv_pad)
            trg_enc_SMI = torch.Tensor(trg_enc_SMI).long()

            #trg_enc_SMI = trg_enc_SMI.permute(1,0)

            ### Load target MF
            src_enc_MF = self._encode_MF(self.stoi_MF, MF)
            src_MF = [self.stoi["<SOS>"]] + src_enc_MF + [self.stoi["<EOS>"]] 
            padd = [0 for i in range(self.config.padding_points_number-len(src_MF))]
            val = [0 for i in range(len(src_MF))]
            src_MF = torch.tensor(src_MF + padd)
            inv_pad = list(map(lambda x: 1 if x == 0 else x, padd))
            mask_MF = torch.tensor(val + inv_pad)

            ### Generate fingerprint for the canonical SMILES
            trg_FP = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fingerprint_size)
            trg_FP = torch.tensor(list(trg_FP), dtype=torch.float32)
            item_dict['src_MF'] = src_MF
            item_dict['mask_MF'] = mask_MF
            item_dict['src_MS'] = src_MS
            item_dict['mask_MS'] = mask_MS        
            item_dict['trg_enc_SMI'] = trg_enc_SMI
            item_dict['trg_FP'] = trg_FP        

        return  item_dict



def collate_fn(batch):
    collated_data = {}
    
    # List of all potential spectrum types
    all_spectrum_types = ['1H', '13C', 'HSQC', 'COSY', 'IR', 'HSQC_', 'COSY_']

    # Iterate over all potential spectrum types and pad sequences if they exist
    for spectrum_type in all_spectrum_types:
        src_key = f'src_{spectrum_type}'
        mask_key = f'mask_{spectrum_type}'
        try:
            src_data = [item[src_key] for item in batch if src_key in item]
            mask_data = [item[mask_key] for item in batch if mask_key in item]
            
            # Only pad if the data exists and is not empty
            if src_data:
                collated_data[src_key] = pad_sequence(src_data, batch_first=True).float()
            if mask_data:
                collated_data[mask_key] = pad_sequence(mask_data, batch_first=True).float()
        except:
            pass

    # Handling MW
    collated_data['trg_MW'] = torch.stack([item['trg_MW'] for item in batch]).float()

    # Handling SMI & FP
    trg_keys = ['trg_enc_SMI']
    for trg_key in trg_keys:
        trg_data = [item[trg_key] for item in batch if trg_key in item]
        if trg_data:
            padded_data = pad_sequence(trg_data, batch_first=True).long()
            padding_size = 64 - padded_data.size(1)
            collated_data[trg_key] = F.pad(padded_data, (0, padding_size))
            #collated_data[trg_key] = pad_sequence(trg_data, batch_first=True).long()

    collated_data['trg_FP'] = pad_sequence([item['trg_FP'] for item in batch], batch_first=True).float()


    # Handling MS
    try: # because optional
        collated_data['src_MS'] = pad_sequence([item['src_MS'] for item in batch], batch_first=True).long()
        collated_data['mask_MS'] = pad_sequence([item['mask_MS'] for item in batch], batch_first=True).long()   
    except:
        pass
    collated_data['src_MF'] = pad_sequence([item['src_MF'] for item in batch], batch_first=True).long()
    collated_data['mask_MF'] = pad_sequence([item['mask_MF'] for item in batch], batch_first=True).long()

    return collated_data
