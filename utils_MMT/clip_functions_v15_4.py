# Standard library imports
import random

# Third-party imports
import rdkit
import torch
import json
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Sampler
from transformers import RobertaModel, RobertaTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import os
import random
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython.display import display, HTML, SVG


# Local module imports
from utils_MMT.dataloaders_pl_v15_4 import collate_fn, MultimodalData
#from utils.models_MMT_v14_1 import MultimodalTransformer
from utils_MMT.models_CLIP_v15_4 import CLIPModel, ChembertaFingerprint, CLIPMultiGPU
import utils_MMT.data_generation_v15_4 as dg

def create_batches(dataset, similarity_threshold, batch_size, max_search_size, weight_delta):
    """
    Create batches of molecules from a dataset based on the Tanimoto similarity and molecular weight.
    
    Args:
        dataset (pd.DataFrame): Dataset containing the molecular information.
        similarity_threshold (float): Minimum Tanimoto similarity to be considered similar.
        batch_size (int): The desired batch size.
        max_search_size (int): The maximum number of molecules to search for each anchor molecule.
        weight_delta (float): The molecular weight range allowed for molecules in the same batch.
        
    Returns:
        batches (list): A list of lists containing indices of molecules in each batch.
    """
    molecules = [Chem.MolFromSmiles(smi) for smi in tqdm(dataset.ref_data["SMILES"])]
    available_indices = set(range(len(molecules)))
    batches = []

    while available_indices:
        current_batch = []
        similar_indices = []

        # Choose a new anchor molecule
        anchor_index = available_indices.pop()
        current_batch.append(anchor_index)
        anchor_mol = molecules[anchor_index]

        # Get the molecular weight range based on the anchor molecule
        anchor_weight = rdkit.Chem.rdMolDescriptors.CalcExactMolWt(anchor_mol)
        min_weight, max_weight = anchor_weight - weight_delta, anchor_weight + weight_delta
        weight_filtered_indices = filter_molecules_by_weight(molecules, min_weight, max_weight)
        # Filter out the unavailable indices
        search_indices = list(set(weight_filtered_indices).intersection(available_indices))
        random.shuffle(search_indices)

        searched_molecules = 0

        for i in tqdm(search_indices):
            if len(current_batch) >= batch_size:
                break
            if tanimoto_similarity(anchor_mol, molecules[i]) >= similarity_threshold:
                current_batch.append(i)
                available_indices.discard(i)

            searched_molecules += 1

            # Stop searching if max_search_size is reached
            if searched_molecules >= max_search_size:
                break

        # Add the remaining spots in the batch with molecules within the weight range
        if len(current_batch) < batch_size:
            for i in search_indices:
                if len(current_batch) >= batch_size:
                    break
                current_batch.append(i)
                available_indices.discard(i)

        # Add the current batch to the list of batches
        batches.append(current_batch)

    return batches

def filter_molecules_by_weight(molecules, min_weight, max_weight):
    """
    Filter molecules based on their molecular weight.
    
    Args:
        molecules (list): A list of RDKit Mol objects.
        min_weight (float): The minimum molecular weight to be included in the filtered list.
        max_weight (float): The maximum molecular weight to be included in the filtered list.
        
    Returns:
        filtered_indices (list): A list of indices corresponding to the molecules with molecular weights within the specified range.
    """
    filtered_indices = []

    for i, mol in enumerate(molecules):
        mol_weight = rdkit.Chem.rdMolDescriptors.CalcExactMolWt(mol)
        if min_weight <= mol_weight <= max_weight:
            filtered_indices.append(i)

    return filtered_indices

class WeightSortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, window_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.window_size = window_size if window_size is not None else max(batch_size, 4)

        # Sort the dataset by molecular weight
        self.sorted_indices = self.dataset.ref_data.sort_values("MW").index.tolist()

    def __iter__(self):
        batch = []
        window = []
        for idx in self.sorted_indices:
            window.append(idx)
            if len(window) == self.window_size:
                random.shuffle(window)
                batch.extend(window)
                window = []

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if len(window) > 0:
            random.shuffle(window)
            batch.extend(window)

        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        else:
            return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size

class TanimotoBatchSampler(Sampler):
    """
    A custom batch sampler for RDKit Mol datasets using Tanimoto similarity and molecular weight for creating batches.
    
    Args:
        dataset (Dataset): A dataset containing RDKit Mol objects.
        batch_size (int): The number of samples per batch.
        drop_last (bool): Whether to drop the last incomplete batch.
        similarity_threshold (float): The minimum Tanimoto similarity threshold for a molecule to be included in the same batch as the anchor molecule.
        max_search_size (int): The maximum number of molecules to search for similarity within a batch.
        weight_delta (float): The range of molecular weights to search for similar molecules around the anchor molecule.
        
    Attributes:
        dataset (Dataset): The dataset containing RDKit Mol objects.
        batch_size (int): The number of samples per batch.
        drop_last (bool): Whether to drop the last incomplete batch.
        batches (list): A list of lists, where each inner list contains indices corresponding to a batch of molecules.
    """
    def __init__(self, dataset, batch_size, drop_last, similarity_threshold, max_search_size, weight_delta):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batches = create_batches(self.dataset, similarity_threshold, batch_size, max_search_size, weight_delta)

    def __iter__(self):
        for batch in self.batches:
            if not self.drop_last or len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        if self.drop_last:
            return sum(1 for b in self.batches if len(b) == self.batch_size)
        else:
            return len(self.batches)


def tanimoto_similarity(mol1, mol2, radius=2, num_class=1024):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=num_class)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=num_class)
    return DataStructs.TanimotoSimilarity(fp1, fp2)



def create_CLIP_dataloaders(config, stoi, stoi_MF):
    #NUM_WORKERS = os.cpu_count()
    batch_size = config.CLIP_batch_size

    train_dataset = MultimodalData(config, 
                                   stoi, 
                                   stoi_MF, 
                                   mode="train")
    
    train_sampler = WeightSortedBatchSampler(train_dataset, 
                                             batch_size, 
                                             drop_last=False)

    test_dataset = MultimodalData(config, 
                                  stoi, 
                                  stoi_MF, 
                                  mode="test")

    test_sampler = WeightSortedBatchSampler(test_dataset, 
                                            batch_size, 
                                            drop_last=False)
    
    val_dataset = MultimodalData(config, 
                                stoi, 
                                stoi_MF, 
                                mode="val")

    #val_sampler = WeightSortedBatchSampler(val_dataset, 
    #                                        1, 
    #                                        drop_last=False)
        
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_sampler=train_sampler,
                                  #shuffle=True, 
                                  collate_fn=collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)
    
    test_dataloader = DataLoader(test_dataset, 
                                  batch_sampler=test_sampler,
                                  #shuffle=False, 
                                  collate_fn=collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1,  # because will run multimodal in parallel on many samples
                                  shuffle=False, 
                                  collate_fn=collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)
    dataloaders = {"train":train_dataloader, "test":test_dataloader, "val":val_dataloader} #, "all":all_dataloader}
    return dataloaders

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]




def run_training_CLIP(train_dataloader, test_dataloader, config):
    if not os.path.exists(config.CLIP_model_save_dir):
        os.mkdir(config.CLIP_model_save_dir)

    # convert the Namespace to dictionary
    config_dict = vars(config)
    # assuming the CLIP_model_save_dir is a list with one element
    config_save_path = config.CLIP_model_save_dir
    # save the dictionary as a JSON file
    random_num = random.randint(0,10000)
    with open(f"{config_save_path}/config_{str(random_num)}.json", 'w') as f:
        json.dump(config_dict, f, indent=4) 

    filepath= config.CLIP_model_save_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        mode='min',
        filepath=os.path.join(filepath,'model_CLIP-{epoch:02d}-{loss:.2f}'),
        save_top_k=-1,  # Keeps all checkpoints.
    )

    # If i use that the saving doesn't work well
    profiler = SimpleProfiler()

    wandb_logger = WandbLogger(project=config.project, 
                               log_model='all') # log all new checkpoints during training
    CLIP_multi_gpu_model = CLIPMultiGPU(config)

    if config.CLIP_continue_training:
        checkpoint_path = config.CLIP_model_path
        CLIP_multi_gpu_model = CLIP_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)

    config_dict = vars(config)
    wandb_logger.log_hyperparams(config_dict)

    try:
        trainer = pl.Trainer(profiler=profiler,
                             gpus=config.gpu_num, 
                             progress_bar_refresh_rate=10, 
                             accelerator='ddp', 
                             logger=wandb_logger,
                             checkpoint_callback=checkpoint_callback,
                             max_epochs=config.CLIP_NUM_EPOCHS,
                             #fast_dev_run=True,
                             #early_stop_callback=early_stopping,
                             #limit_train_batches=1,
                             #limit_val_batches=1
                             )
        trainer.fit(CLIP_multi_gpu_model, train_dataloader, test_dataloader)
    except Exception as e:
        print(f"Error occurred: {e}")
        backup_ckpt_path = config.CLIP_model_save_dir+"/last_backup_checkpoint.ckpt"
        trainer.save_checkpoint(backup_ckpt_path)
        print("Model saved.")

def run_CLIP(config, stoi, stoi_MF):
    dataloaders = create_CLIP_dataloaders(config, stoi, stoi_MF)

    #train_dataloader, test_dataloader = load_dataloaders(config, stoi, stoi_MF)
    run_training_CLIP(dataloaders["train"], dataloaders["test"], config)
   
    
########### Decide if I want to use  dg.create_CLIP_dataloaders or from here ###########