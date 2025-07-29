from __future__ import annotations

from typing import TYPE_CHECKING


import time
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from pysmilesutils.tokenize import SMILESTokenizer
from rdkit import Chem, RDLogger

from ..scores import ScoreCollection
from .. import smiles_utils

if TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple
    from ...models.base_transformer import _AbsTransformerModel

RDLogger.DisableLog("rdApp.*")

class MultinomialSearchSampler:
    """
    GPU-optimized multinomial search sampler/decoder. Generates predictions and
    calculates performance metrics.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        scorers: ScoreCollection,
        max_sequence_length: int,
        device: str = "cuda",
        data_device: str = "cuda",
        sample_unique: bool = True,
    ) -> None:
        """
        Args:
            tokenizer (SMILESTokenizer): Tokenizer with vocabulary.
            max_sequence_length (int): Maximum generated sequence length.
            device (str): "cuda" or "cpu".
            data_device (str): device used for handling the data. If memory issues,
                could help to set data_device="cpu"
            sampled_unique (bool):  Whether to return unique multinomial search solutions.
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.smiles_unique = None
        self.log_lhs_unique = None
        self.sampling_alg = None
        self.data_device = data_device
        self.sample_unique = sample_unique
        self.scorers = scorers

        self.begin_token_id = self.tokenizer["start"]
        self.pad_token_id = self.tokenizer["pad"]
        self.end_token_id = self.tokenizer["end"]


    @torch.no_grad()
    def sample_molecules(
        self,
        model: _AbsTransformerModel,
        batch_input: Dict[str, Any],
        num_samples: int,
        sampling_alg: str = "multinomial",
        return_tokenized: bool = False,
        temperature: float = 1.0,
        max_attempts: int = 30,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Sample unique molecules from the model using multinomial sampling with temperature adjustment.

        Args:
            model (_AbsTransformerModel): The transformer base model (e.g. BARTModel or UnifiedModel)
            batch_input (Dict): The input, X, to the network
            num_samples (int): Number of unique samples to generate for each input
            sampling_alg (str): Sampling algorithm (only "multinomial" is supported)
            return_tokenized (bool): Whether to return the sampled tokens (True), or the converted SMILES (False)
            temperature (float): Initial temperature for multinomial sampling
            max_attempts (int): Maximum number of sampling attempts before giving up

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Two lists of numpy arrays, 
                the first containing num_samples unique SMILES strings for each input,
                the second containing the corresponding log probabilities
        """
        print("DEBUG: Entering MultinomialSearchSampler.sample_molecules()")
        print(f"DEBUG: num_samples={num_samples}, sampling_alg={sampling_alg}, temperature={temperature}")
        print(f"DEBUG: max_attempts={max_attempts}, return_tokenized={return_tokenized}")
        self.sampling_alg = sampling_alg
        print("sampler MNS")

        if self.device is None:
            self.device = next(model.parameters()).device
        print(f"DEBUG: Using device: {self.device}")

        if sampling_alg != "multinomial":
            print(f"DEBUG: Error - sampling_alg={sampling_alg} not supported")
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        start_time = time.time()
        _, batch_size = tuple(batch_input["encoder_input"].shape)
        print(f"DEBUG: Batch size: {batch_size}")
        memory = model.encode(batch_input)
        print(f"DEBUG: Encoding time: {time.time() - start_time:.2f} seconds")
        print(f"DEBUG: Memory shape: {memory.shape}")

        all_sampled_smiles = []
        all_log_probs = []

        for batch_idx in range(batch_size):
            print(f"DEBUG: Processing batch item {batch_idx+1}/{batch_size}")
            unique_smiles = set()
            unique_mols = set()
            sample_log_probs = []
            current_temperature = temperature
            attempts = 0
            dup_dim = 16  # 64
            while len(unique_smiles) < num_samples and attempts < max_attempts:
                print(f"DEBUG: Attempt {attempts+1}/{max_attempts}, unique molecules so far: {len(unique_smiles)}/{num_samples}")
                print(f"DEBUG: Current temperature: {current_temperature}")
                attempts += 1
                loop_start_time = time.time()

                # Duplicate the memory for this batch item
                batch_memory = memory[:, batch_idx:batch_idx+1].repeat(1, dup_dim, 1)
                batch_padding_mask = batch_input["encoder_pad_mask"][:,batch_idx:batch_idx+1].repeat(1,dup_dim)

                token_ids = torch.full((1, dup_dim), self.begin_token_id, dtype=torch.long, device=self.device)
                pad_mask = torch.zeros_like(token_ids, dtype=torch.bool, device=self.device)

                sampled_tokens = []
                log_probs = torch.zeros(dup_dim, device=self.device)
                
                sequence_ended = torch.zeros(dup_dim, dtype=torch.bool, device=self.device)
                print(f"DEBUG: Starting token generation loop")
                
                for step in range(self.max_sequence_length - 1):
                    decode_input = {
                        "decoder_input": token_ids,
                        "decoder_pad_mask": pad_mask,
                        "memory_input": batch_memory,
                        "memory_pad_mask": batch_padding_mask,
                    }

                    logits = model.decode(decode_input)
                    logits = logits[-1, :, :] / current_temperature
                    probs = F.softmax(logits, dim=-1)

                    active_seq = ~sequence_ended
                    next_tokens = torch.full((dup_dim,), self.pad_token_id, dtype=torch.long, device=self.device)
                    next_tokens[active_seq] = torch.multinomial(probs[active_seq], num_samples=1).squeeze(-1)
                    sampled_tokens.append(next_tokens)

                    log_probs[active_seq] += F.log_softmax(logits[active_seq], dim=-1).gather(1, next_tokens[active_seq].unsqueeze(-1)).squeeze(-1)

                    token_ids = torch.cat([token_ids, next_tokens.unsqueeze(0)], dim=0)
                    pad_mask = torch.cat([pad_mask, sequence_ended.unsqueeze(0)], dim=0)

                    sequence_ended = sequence_ended | (next_tokens == self.end_token_id)
                    
                    # Log progress every 10 steps
                    if step % 10 == 0:
                        print(f"DEBUG: Generation step {step}, {sequence_ended.sum().item()}/{dup_dim} sequences finished")

                    if sequence_ended.all():
                        print(f"DEBUG: All sequences finished at step {step}")
                        break
                
                print(f"DEBUG: Token generation complete in {time.time() - loop_start_time:.2f} seconds")
                sampled_tokens = torch.stack(sampled_tokens).transpose(0, 1)
                
                print(f"DEBUG: Converting tokens to SMILES")
                # Convert tokens to SMILES
                if return_tokenized:
                    batch_smiles = sampled_tokens.cpu().numpy()
                else:
                    tokens = self.tokenizer.convert_ids_to_tokens(sampled_tokens)
                    batch_smiles = self.tokenizer.detokenize(tokens, truncate_at_end_token=True)

                # Process each SMILES in the batch
                for smiles, prob in zip(batch_smiles, log_probs):
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            print(smiles)
                            canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                            if canon_smiles not in unique_smiles:
                                unique_smiles.add(canon_smiles)
                                unique_mols.add(mol)
                                sample_log_probs.append(prob.item())
                    except:
                        continue  # Invalid SMILES, skip

                print(f"Batch {batch_idx + 1}, Attempt {attempts}, Unique molecules: {len(unique_smiles)}, Time: {time.time() - loop_start_time:.2f} seconds")

                if len(unique_smiles) < num_samples:
                    current_temperature += 0.1
                    print(f"Increasing temperature to {current_temperature}")
                print(attempts, len(unique_smiles))
                #import IPython; IPython.embed()

            # Truncate or pad the results to num_samples
            unique_smiles_list = list(unique_smiles)[:num_samples]
            sample_log_probs = sample_log_probs[:num_samples]

            # Pad if we don't have enough samples
            while len(unique_smiles_list) < num_samples:
                unique_smiles_list.append("")
                sample_log_probs.append(float('-inf'))

            print(f"DEBUG: Final unique SMILES for batch {batch_idx+1}: {len(unique_smiles_list)}")
            if len(unique_smiles_list) > 0:
                print(f"DEBUG: First SMILES: {unique_smiles_list[0]}")
            
            all_sampled_smiles.append(np.array(unique_smiles_list))
            all_log_probs.append(np.array(sample_log_probs))

        print(f"DEBUG: Final results structure:")
        print(f"DEBUG: all_sampled_smiles length: {len(all_sampled_smiles)}")
        for i, smiles_array in enumerate(all_sampled_smiles):
            print(f"DEBUG: Batch {i+1} has {len(smiles_array)} SMILES")
        
        print(f"DEBUG: all_log_probs length: {len(all_log_probs)}")
        for i, probs_array in enumerate(all_log_probs):
            print(f"DEBUG: Batch {i+1} has {len(probs_array)} log probs")
        
        # Store the results for potential use in other methods
        self.smiles_unique = all_sampled_smiles
        self.log_lhs_unique = all_log_probs
        
        print(f"Total sampling time: {time.time() - start_time:.2f} seconds")
        return all_sampled_smiles, all_log_probs



    # @torch.no_grad()
    # def sample_molecules(
    #     self,
    #     model: _AbsTransformerModel,
    #     batch_input: Dict[str, Any],
    #     num_samples: int,
    #     sampling_alg: str = "multinomial",
    #     return_tokenized: bool = False,
    #     temperature: float = 1.0,
    # ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    #     """Sample molecules from the model using multinomial sampling with early stopping.

    #     Args:
    #         model (_AbsTransformerModel): The transformer base model (e.g. BARTModel or UnifiedModel)
    #         batch_input (Dict): The input, X, to the network
    #         num_samples (int): Number of samples to generate for each input
    #         sampling_alg (str): Sampling algorithm (only "multinomial" is supported)
    #         return_tokenized (bool): Whether to return the sampled tokens (True), or the converted SMILES (False)
    #         temperature (float): Temperature for multinomial sampling

    #     Returns:
    #         Tuple[List[np.ndarray], List[np.ndarray]]: Two lists of numpy arrays, 
    #             the first containing num_samples SMILES strings for each input,
    #             the second containing the corresponding log probabilities
    #     """
    #     self.sampling_alg = sampling_alg
    #     print("sampler MNS")

    #     if self.device is None:
    #         self.device = next(model.parameters()).device

    #     print(sampling_alg)
    #     if sampling_alg != "multinomial":
    #         raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

    #     start_time = time.time()
    #     _, batch_size = tuple(batch_input["encoder_input"].shape)
    #     memory = model.encode(batch_input)
    #     print(f"Encoding time: {time.time() - start_time:.2f} seconds")

    #     all_sampled_smiles = []
    #     all_log_probs = []
    #     for i in range(num_samples):
    #         loop_start_time = time.time()
    #         token_ids = torch.full((1, batch_size), self.begin_token_id, dtype=torch.long, device=self.device)
    #         pad_mask = torch.zeros_like(token_ids, dtype=torch.bool, device=self.device)

    #         sampled_tokens = []
    #         log_probs = torch.zeros(batch_size, device=self.device)
            
    #         # Keep track of which sequences have ended
    #         sequence_ended = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    #         for step in range(self.max_sequence_length - 1):
    #             step_start_time = time.time()
    #             decode_input = {
    #                 "decoder_input": token_ids,
    #                 "decoder_pad_mask": pad_mask,
    #                 "memory_input": memory,
    #                 "memory_pad_mask": batch_input["encoder_pad_mask"],
    #             }
    #             import IPython; IPython.embed()

    #             logits = model.decode(decode_input)
    #             logits = logits[-1, :, :] / temperature
    #             probs = F.softmax(logits, dim=-1)

    #             # Only sample for sequences that haven't ended
    #             active_seq = ~sequence_ended
    #             next_tokens = torch.full((batch_size,), self.pad_token_id, dtype=torch.long, device=self.device)
    #             next_tokens[active_seq] = torch.multinomial(probs[active_seq], num_samples=1).squeeze(-1)
    #             sampled_tokens.append(next_tokens)

    #             # Update log probabilities only for active sequences
    #             log_probs[active_seq] += F.log_softmax(logits[active_seq], dim=-1).gather(1, next_tokens[active_seq].unsqueeze(-1)).squeeze(-1)

    #             token_ids = torch.cat([token_ids, next_tokens.unsqueeze(0)], dim=0)
    #             pad_mask = torch.cat([pad_mask, sequence_ended.unsqueeze(0)], dim=0)

    #             # Update which sequences have ended
    #             sequence_ended = sequence_ended | (next_tokens == self.end_token_id)

    #             if sequence_ended.all():
    #                 break

    #         print(f"Sample {i + 1} generation time: {time.time() - loop_start_time:.2f} seconds")

    #         sampled_tokens = torch.stack(sampled_tokens).transpose(0, 1)

    #         if return_tokenized:
    #             all_sampled_smiles.append(sampled_tokens.cpu().numpy())
    #         else:
    #             tokens = self.tokenizer.convert_ids_to_tokens(sampled_tokens)
    #             sampled_smiles = np.asarray(self.tokenizer.detokenize(tokens, truncate_at_end_token=True))
    #             all_sampled_smiles.append(sampled_smiles)

    #         all_log_probs.append(log_probs.cpu().numpy())

    #     # Reshape the results to match the desired output format
    #     result_smiles = [np.array(sample) for sample in zip(*all_sampled_smiles)]
    #     result_log_probs = [np.array(sample) for sample in zip(*all_log_probs)]

    #     print(f"Total sampling time: {time.time() - start_time:.2f} seconds")
    #     return result_smiles, result_log_probs






    # @torch.no_grad()
    # def sample_molecules(
    #     self,
    #     model: _AbsTransformerModel,
    #     batch_input: Dict[str, Any],
    #     num_samples: int,
    #     sampling_alg: str = "multinomial",
    #     return_tokenized: bool = False,
    #     temperature: float = 1.0,
    # ) -> List[np.ndarray]:
    #     """Sample molecules from the model using multinomial sampling.

    #     Args:
    #         model (_AbsTransformerModel): The transformer base model (e.g. BARTModel or UnifiedModel)
    #         batch_input (Dict): The input, X, to the network
    #         num_samples (int): Number of samples to generate for each input
    #         sampling_alg (str): Sampling algorithm (only "multinomial" is supported)
    #         return_tokenized (bool): Whether to return the sampled tokens (True), or the converted SMILES (False)
    #         temperature (float): Temperature for multinomial sampling

    #     Returns:
    #         List[np.ndarray]: List of numpy arrays, each containing num_samples SMILES strings for each input
    #     """
    #     self.sampling_alg = sampling_alg
    #     print("sampler MNS")

    #     if self.device is None:
    #         self.device = next(model.parameters()).device

    #     print(sampling_alg)
    #     if sampling_alg != "multinomial":
    #         raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

    #     start_time = time.time()
    #     _, batch_size = tuple(batch_input["encoder_input"].shape)
    #     memory = model.encode(batch_input)
    #     print(f"Encoding time: {time.time() - start_time:.2f} seconds")

    #     all_sampled_smiles = []
    #     all_log_probs = []
    #     for i in range(num_samples):
    #         loop_start_time = time.time()
    #         token_ids = torch.full((1, batch_size), self.begin_token_id, dtype=torch.long, device=self.device)
    #         pad_mask = torch.zeros_like(token_ids, dtype=torch.bool, device=self.device)

    #         sampled_tokens = []
    #         log_probs = torch.zeros(batch_size, device=self.device)
    #         for step in range(self.max_sequence_length - 1):
    #             step_start_time = time.time()
    #             decode_input = {
    #                 "decoder_input": token_ids,
    #                 "decoder_pad_mask": pad_mask,
    #                 "memory_input": memory,
    #                 "memory_pad_mask": batch_input["encoder_pad_mask"],
    #             }

    #             logits = model.decode(decode_input)
    #             logits = logits[-1, :, :] / temperature
    #             probs = F.softmax(logits, dim=-1)

    #             next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    #             sampled_tokens.append(next_tokens)

    #             log_probs += F.log_softmax(logits, dim=-1).gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)

    #             token_ids = torch.cat([token_ids, next_tokens.unsqueeze(0)], dim=0)
    #             pad_mask = torch.cat([pad_mask, torch.zeros_like(next_tokens, dtype=torch.bool).unsqueeze(0)], dim=0)

    #             if (next_tokens == self.end_token_id).all():
    #                 break

    #             #print(f"Step {step + 1} time: {time.time() - step_start_time:.2f} seconds")
    #         print(f"Sample {i + 1} generation time: {time.time() - loop_start_time:.2f} seconds")
    #         import IPython; IPython.embed()

    #         sampled_tokens = torch.stack(sampled_tokens).transpose(0, 1)

    #         if return_tokenized:
    #             all_sampled_smiles.append(sampled_tokens.cpu().numpy())
    #         else:
    #             tokens = self.tokenizer.convert_ids_to_tokens(sampled_tokens)
    #             sampled_smiles = np.asarray(self.tokenizer.detokenize(tokens, truncate_at_end_token=True))
    #             all_sampled_smiles.append(sampled_smiles)

    #         all_log_probs.append(log_probs.cpu().numpy())

    #     # Reshape the results to match the desired output format
    #     result_smiles = [np.array(sample) for sample in zip(*all_sampled_smiles)]
    #     result_log_probs = [np.array(sample) for sample in zip(*all_log_probs)]

    #     print(f"Total sampling time: {time.time() - start_time:.2f} seconds")
    #     return result_smiles, result_log_probs




    # @torch.no_grad()
    # def sample_molecules(
    #     self,
    #     model: _AbsTransformerModel,
    #     batch_input: Dict[str, Any],
    #     num_samples: int,
    #     sampling_alg: str = "multinomial",
    #     return_tokenized: bool = False,
    #     temperature: float = 1.0,
    # ) -> List[np.ndarray]:
    #     """Sample molecules from the model using multinomial sampling.

    #     Args:
    #         model (_AbsTransformerModel): The transformer base model (e.g. BARTModel or UnifiedModel)
    #         batch_input (Dict): The input, X, to the network
    #         num_samples (int): Number of samples to generate for each input
    #         sampling_alg (str): Sampling algorithm (only "multinomial" is supported)
    #         return_tokenized (bool): Whether to return the sampled tokens (True), or the converted SMILES (False)
    #         temperature (float): Temperature for multinomial sampling

    #     Returns:
    #         List[np.ndarray]: List of numpy arrays, each containing num_samples SMILES strings for each input
    #     """
    #     self.sampling_alg = sampling_alg
    #     print("sampler MNS")

    #     if self.device is None:
    #         self.device = next(model.parameters()).device

    #     print(sampling_alg)
    #     if sampling_alg != "multinomial":
    #         raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

    #     _, batch_size = tuple(batch_input["encoder_input"].shape)
    #     memory = model.encode(batch_input)

    #     all_sampled_smiles = []
    #     all_log_probs = []
    #     for _ in range(num_samples):
    #         token_ids = torch.full((1, batch_size), self.begin_token_id, dtype=torch.long, device=self.device)
    #         pad_mask = torch.zeros_like(token_ids, dtype=torch.bool, device=self.device)

    #         sampled_tokens = []
    #         log_probs = torch.zeros(batch_size, device=self.device)
    #         for _ in range(self.max_sequence_length - 1):
    #             decode_input = {
    #                 "decoder_input": token_ids,
    #                 "decoder_pad_mask": pad_mask,
    #                 "memory_input": memory,
    #                 "memory_pad_mask": batch_input["encoder_pad_mask"],
    #             }      

    #             logits = model.decode(decode_input)
    #             logits = logits[-1, :, :] / temperature
    #             probs = F.softmax(logits, dim=-1)

    #             next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    #             sampled_tokens.append(next_tokens)

    #             log_probs += F.log_softmax(logits, dim=-1).gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)

    #             token_ids = torch.cat([token_ids, next_tokens.unsqueeze(0)], dim=0)
    #             pad_mask = torch.cat([pad_mask, torch.zeros_like(next_tokens, dtype=torch.bool).unsqueeze(0)], dim=0)

    #             if (next_tokens == self.end_token_id).all():
    #                 break
    #         print("break")
    #         import IPython; IPython.embed()

    #         sampled_tokens = torch.stack(sampled_tokens).transpose(0, 1)

    #         if return_tokenized:
    #             all_sampled_smiles.append(sampled_tokens.cpu().numpy())
    #         else:
    #             tokens = self.tokenizer.convert_ids_to_tokens(sampled_tokens)
    #             sampled_smiles = np.asarray(self.tokenizer.detokenize(tokens, truncate_at_end_token=True))
    #             all_sampled_smiles.append(sampled_smiles)
            
    #         all_log_probs.append(log_probs.cpu().numpy())

    #     # Reshape the results to match the desired output format
    #     result_smiles = [np.array(sample) for sample in zip(*all_sampled_smiles)]
    #     result_log_probs = [np.array(sample) for sample in zip(*all_log_probs)]

    #     return result_smiles, result_log_probs

    # @torch.no_grad()
    # def sample_molecules(
    #     self,
    #     model: _AbsTransformerModel,
    #     batch_input: Dict[str, Any],
    #     num_samples: int,
    #     sampling_alg: str = "multinomial",
    #     return_tokenized: bool = False,
    #     temperature: float = 1.0,
    #  ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Sample molecules from the model using multinomial sampling.

    #     Args:
    #         model (_AbsTransformerModel): The transformer base model (e.g. BARTModel or
    #             UnifiedModel)
    #         batch_input (Dict): The input, X, to the network
    #         num_samples (int): Number of samples to generate for each input
    #         temperature (float): Temperature for multinomial sampling
    #         return_tokenized: whether to return the sampled tokens (True), or the
    #             converted SMILES (False). Defaults to False.

    #     Returns:
    #         (SMILES of sampled molecules and log-likelihoods) or
    #         (token indices of sampled molecules and log-likelihoods)
    #     """
    #     self.sampling_alg = sampling_alg
    #     print("sampler MNS")

    #     if self.device is None:
    #         self.device = next(model.parameters()).device

    #     _, batch_size = tuple(batch_input["encoder_input"].shape)

    #     if sampling_alg != "multinomial":
    #         raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

    #     token_ids = torch.full((1, batch_size), self.begin_token_id, dtype=torch.long, device=self.device)
    #     pad_mask = torch.zeros_like(token_ids, dtype=torch.bool, device=self.device)

    #     sampled_tokens = []
    #     log_probs = torch.zeros(batch_size, device=self.device)

    #     memory = model.encode(batch_input)

    #     for _ in range(self.max_sequence_length - 1):
    #         decode_input = {
    #             "decoder_input": token_ids,
    #             "decoder_pad_mask": pad_mask,
    #             "memory_input": memory,
    #             "memory_pad_mask": batch_input["encoder_pad_mask"],
    #             }      

    #         logits = model.decode(decode_input)
    #         logits = logits[-1, :, :] / temperature
    #         probs = F.softmax(logits, dim=-1)


    #         next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    #         sampled_tokens.append(next_tokens)

    #         log_probs += F.log_softmax(logits, dim=-1).gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)

    #         token_ids = torch.cat([token_ids, next_tokens.unsqueeze(0)], dim=0)
    #         pad_mask = torch.cat([pad_mask, torch.zeros_like(next_tokens, dtype=torch.bool).unsqueeze(0)], dim=0)

    #         if (next_tokens == self.end_token_id).all():
    #             break

    #     sampled_tokens = torch.stack(sampled_tokens).transpose(0, 1)

    #     if return_tokenized:
    #         return sampled_tokens.cpu().numpy(), log_probs.cpu().numpy()
    #     else:
    #         tokens = self.tokenizer.convert_ids_to_tokens(sampled_tokens)
    #         sampled_smiles = np.asarray(self.tokenizer.detokenize(tokens, truncate_at_end_token=True))

    #         if self.sample_unique:
    #             self.smiles_unique, self.log_lhs_unique = smiles_utils.uniqueify_sampled_smiles(
    #                 sampled_smiles, log_probs.cpu().numpy(), num_samples
    #             )
    #             return self.smiles_unique, self.log_lhs_unique
    #         else:
                
    #             #import IPython; IPython.embed()

    #             return sampled_smiles, log_probs.cpu().numpy()





    def compute_sampling_metrics(
        self,
        sampled_smiles: List[np.ndarray],
        target_smiles: List[str],
        is_canonical: bool = False,
    ) -> Dict[str, Any]:
        n_samples = len(sampled_smiles)
        n_targets = len(target_smiles)
        err_msg = f"The number of sampled and target molecules must be the same, got {n_samples} and {n_targets}"
        assert n_samples == n_targets, err_msg

        if is_canonical:
            for scorer in self.scorers.objects():
                if not getattr(scorer, "canonicalized", True):
                    setattr(scorer, "canonicalized", True)
                    print("Configuring scorers for pre-canonicalized SMILES.")

        metrics = self.scorers.score(sampled_smiles, target_smiles)
        return metrics

        
    def compute_sampling_metrics(
        self,
        sampled_smiles: List[np.ndarray],
        target_smiles: List[str],
        is_canonical: bool = False,
    ) -> Dict[str, Any]:
        """
        Uses a ScoreCollection to evaluated a list of sampled_smiles given target_smiles.
        Compute sampling metrics given sampled SMILES and target SMILES.
        Computes the scores loaded in self.scorers (ScoreCollection).

        Args:
            sampled_smiles: list of top-k sampled SMILES
            target_smiles: list of target SMILES
            is_canonical: If True, will skip canonicalization
        """
        n_samples = len(sampled_smiles)
        n_targets = len(target_smiles)
        err_msg = f"The number of sampled and target molecules must be the same, got {n_samples} and {n_targets}"
        assert n_samples == n_targets, err_msg

        if is_canonical:
            for scorer in self.scorers.objects():
                if not getattr(scorer, "canonicalized", True):
                    setattr(scorer, "canonicalized", True)
                    print("Configuring scorers for pre-canonicalized SMILES.")

        metrics = self.scorers.score(sampled_smiles, target_smiles)
        return metrics
    @staticmethod
    def _calc_multinomial_metrics(sampled_smiles, target_smiles):
        sampled_mols = [[Chem.MolFromSmiles(smi) for smi in smiles] for smiles in sampled_smiles]
        invalid = [[mol is None for mol in mols] for mols in sampled_mols]

        canon_smiles = [["Unknown" if mol is None else Chem.MolToSmiles(mol) for mol in mols] for mols in sampled_mols]
        correct_smiles = [[target_smiles[idx] == smi for smi in smiles] for idx, smiles in enumerate(canon_smiles)]

        num_correct = sum(any(correct) for correct in correct_smiles)
        total = len(correct_smiles)
        num_invalid = sum(sum(inv) for inv in invalid)
        total_samples = sum(len(smiles) for smiles in sampled_smiles)
        perc_invalid = num_invalid / total_samples
        accuracy = num_correct / total

        metrics = {"accuracy": accuracy, "fraction_invalid": perc_invalid}

        return metrics