from interprot.interprot.sae_model import SparseAutoencoder
import torch
from pathlib import Path
import pandas as pd

def get_mutations_str(wt_seq: str, mut_seq: str) -> str:
    """Helper function to format mutations as a string like 'A123B'."""
    mutations = [
        f"{wt_aa}{i+1}{mut_aa}"
        for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq))
        if wt_aa != mut_aa
    ]
    return ":".join(mutations) if mutations else "No mutations"


def get_logits_embeddings(seqs, esm_model, tokenizer, device, batch_size=64, embedding_dim=None):
    """
    Extract mean logit embeddings from sequences using an ESM model.

    Args:
        seqs (List[str]): List of protein sequences.
        esm_model: Loaded ESM model.
        tokenizer: ESM tokenizer.
        device: Device to use for computation.
        batch_size (int): Number of sequences to process in each batch.
        embedding_dim (int): The dimension of the final embedding (model's vocabulary size).

    Returns:
        np.ndarray: Embedding matrix of shape (n_sequences, embedding_dim).
    """
    if embedding_dim is None:
        embedding_dim = esm_model.config.vocab_size
    out_NV = [] # list length N, each entry is V dimensional
    if device.type=="cuda": torch.cuda.empty_cache()
    for start in tqdm(range(0,len(seqs),batch_size),desc="Computing logits",leave=False):
        batch_B = seqs[start:start+batch_size]
        inputs_BT = tokenizer(batch_B, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            logits_BTV = esm_model(**inputs_BT).logits
        for i in range(logits_BTV.size(0)): # loop length B
            seq_logits_TV = logits_BTV[i,1:-1] # drop CLS/EOS
            out_NV.append(seq_logits_TV.mean(0).cpu().numpy() if seq_logits_TV.numel() else
                       torch.zeros(embedding_dim).numpy())
    return np.vstack(out_NV)

def apply_mutations(wt_seq, mutation_str):
    seq = list(wt_seq)
    if pd.isna(mutation_str) or mutation_str == "":
        return wt_seq
    for mut in mutation_str.split(":"):
        pos = int(mut[1:-1]) - 1
        new_residue = mut[-1]
        seq[pos] = new_residue
    return "".join(seq)


def load_sae_model(sae_path, plm_dim, sae_dim, device):
    """Load SAE model from path following probe_esm.py pattern"""
    if not sae_path or not Path(sae_path).exists():
        return None
    
    sae_model = SparseAutoencoder(plm_dim, sae_dim).to(device).eval()
    checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae_state_dict = checkpoint.get('state_dict', checkpoint)
    expected_keys = {'w_enc', 'b_enc', 'w_dec', 'b_pre'}
    prefixes_to_try = ["sae_model.", "sae.", "model.autoencoder.", "autoencoder.", "model.", ""]
    for prefix in prefixes_to_try:
        candidate_dict = sae_state_dict
        if prefix:
            if all(isinstance(k, str) and k.startswith(prefix) for k in sae_state_dict.keys()):
                candidate_dict = {k[len(prefix):]: v for k, v in sae_state_dict.items()}
            else:
                candidate_dict = None
        if candidate_dict and expected_keys.issubset(candidate_dict.keys()):
            sae_model.load_state_dict(candidate_dict, strict=True)
            return sae_model
    sae_model.load_state_dict(sae_state_dict, strict=True)
    return sae_model


import numpy as np
import pandas as pd
from collections import defaultdict
import os
import torch
import tqdm

class MSA_processing:
    def __init__(self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=0.3,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True
        ):
        
        """
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corespondding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that; 
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        """
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols

        self.gen_alignment()
        self.create_all_singles()

    def gen_alignment(self):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i==0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line

        
        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up
            msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".","-")).apply(lambda x: ''.join([aa.upper() for aa in x]))
            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = [aa!='-' for aa in msa_df.sequence[self.focus_seq_name]]
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa for aa,non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
            assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0,"Invalid fragment filtering parameter"
            assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0,"Invalid focus position filtering parameter"
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = np.array(list(map(lambda seq: [aa=='-' for aa in seq], msa_array)))
            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps
            print("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            print("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
            # Lower case non focus cols and filter fragment sequences
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in zip(x, index_cols_below_threshold)]))
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            for seq_idx in range(len(msa_df['sequence'])):
                self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        focus_loc = self.focus_seq_name.split("/")[-1]
        start,stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 

        # Move all letters to CAPS; keeps focus columns only
        for seq_name,sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".","-")
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name,sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        print ("Encoding sequences")
        self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict: 
                    k = self.aa_dict[letter]
                    self.one_hot_encoding[i,j,k] = 1.0

        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                print("Loaded sequence weights from disk")
            except:
                print ("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))
                def compute_weight(seq):
                    number_non_empty_positions = np.dot(seq,seq)
                    if number_non_empty_positions>0:
                        denom = np.dot(list_seq,seq) / np.dot(seq,seq) 
                        denom = np.sum(denom > 1 - self.theta) 
                        return 1/denom
                    else:
                        return 0.0 #return 0 weight if sequence is fully empty
                self.weights = np.array(list(map(compute_weight,list_seq)))
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]

        print ("Neff =",str(self.Neff))
        print ("Data Shape =",self.one_hot_encoding.shape)
    
    def create_all_singles(self):
        start_idx = self.focus_start_loc
        focus_seq_index = 0
        self.mutant_to_letter_pos_idx_focus_list = {}
        list_valid_mutations = []
        # find all possible valid mutations that can be run with this alignment
        alphabet_set = set(list(self.alphabet))
        for i,letter in enumerate(self.focus_seq):
            if letter in alphabet_set and letter != "-":
                for mut in self.alphabet:
                    pos = start_idx+i
                    if mut != letter:
                        mutant = letter+str(pos)+mut
                        self.mutant_to_letter_pos_idx_focus_list[mutant] = [letter, pos, focus_seq_index]
                        list_valid_mutations.append(mutant)
                focus_seq_index += 1   
        self.all_single_mutations = list_valid_mutations

    def save_all_singles(self, output_filename):
        with open(output_filename, "w") as output:
            output.write('mutations')
            for mutation in self.all_single_mutations:
                output.write('\n')
                output.write(mutation)