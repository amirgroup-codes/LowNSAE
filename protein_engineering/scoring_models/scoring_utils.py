import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import pickle
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Typing suffixes:
# L: Total number of amino acids, in this case 20
# T: Sequence length of protein (variable)
# N: Number of sequences (variable)
# E: Length of one-hot encoded sequence (L * T)
# B: Batch size
# V: LM vocabulary size
# H: Embedding dimension of ESM
# D: Embedding dimension of SAE
# ──────────────────────────────────────────────────────────────────────────────

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def one_hot_encode_sequence(seq: str):
    """
    One-hot encodes an entire amino acid sequence.
    Input: String length T
    Output: Vector length L*T
    """
    encoding_LT = np.zeros((len(seq), len(AA_LIST)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            encoding_LT[i, AA_TO_IDX[aa]] = 1.0
    return encoding_LT.flatten()

def score_mlp(sequences: list[str], model: torch.nn.Module):
    """
    Scores a list of sequences using a trained MLP model.
    The model is assumed to have been trained on full-sequence one-hot encodings.

    Args:
        sequences (List[str]): A list of protein sequences.
        model (torch.nn.Module): The trained PyTorch model.

    Returns:
        np.ndarray: An array of predicted scores length N.
    """
    if not sequences:
        return np.array([])

    # One-hot encode each full sequence
    X_NE = np.stack([one_hot_encode_sequence(seq) for seq in sequences])
    X_tensor_NE = torch.tensor(X_NE, dtype=torch.float32).to(next(model.parameters()).device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor_NE).squeeze().cpu().numpy()
        
    return preds



"""
Utils for linear probe
"""
def load_probe_model(model_path):
    """
    Load a trained linear probe model from a pickle file.
    
    Args:
        model_path (str or Path): Path to the pickle file containing the probe model
        
    Returns:
        tuple: (ridge_model, scaler) where ridge_model is the trained Ridge regression model
               and scaler is the fitted StandardScaler
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        ridge_model = model_data['ridge_model']
        scaler = model_data['scaler']
    else:
        # Handle case where only the ridge model was saved
        ridge_model = model_data
        scaler = None
    
    return ridge_model, scaler


def score_probe_model(sequences, probe_model, tokenizer, esm_model, sae_model=None, batch_size=64, layer=24, embedding_dim=1280, logits=False):
    """
    Score sequences using a preloaded probe model and optionally SAE embeddings.
    
    Args:
        sequences (List[str]): Sequences to score.
        probe_model (Tuple): (ridge_model, scaler)
        tokenizer: ESM tokenizer
        esm_model: Preloaded ESM model
        sae_model: Preloaded SAE model or None
    
    Returns:
        np.ndarray: 1D array of predicted scores.
    """
    ridge_model, scaler = probe_model
    if logits:
        embeddings = get_logits_embeddings(sequences, esm_model, tokenizer, batch_size=batch_size, embedding_dim=embedding_dim)
    else:
        embeddings = get_embeddings(sequences, esm_model, tokenizer, sae_model, batch_size=batch_size, layer=layer, embedding_dim=embedding_dim)
    if scaler is not None:
        embeddings = scaler.transform(embeddings)
    
    predictions = ridge_model.predict(embeddings)
    return predictions


def get_embeddings(sequences, esm_model, tokenizer, sae_model=None, batch_size=64, layer=24, embedding_dim=1280):
    """
    Extract embeddings from sequences using ESM model and optionally SAE.
    
    Args:
        sequences (List[str]): List of protein sequences
        esm_model: Loaded ESM model
        sae_model: Loaded SAE model (can be None)
        tokenizer: ESM tokenizer
        
    Returns:
        np.ndarray: Embedding matrix of shape (n_sequences, embedding_dim)
    """
    import sys
    sys.path.append('../..') 
    from interprot.interprot.utils import get_layer_activations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    embeddings_N = []
    for start_idx in range(0, len(sequences), batch_size):
        batch_sequences_B = sequences[start_idx:start_idx + batch_size]
        with torch.inference_mode():
            batch_layer_activations_BTH = get_layer_activations(
                tokenizer, esm_model, batch_sequences_B, layer=layer, device=device
            )
        batch_embeddings_B = []
        for single_sequence_activations_TH in batch_layer_activations_BTH:
            core_activations_TH = single_sequence_activations_TH[1:-1].to(device)
            if core_activations_TH.numel() == 0:
                batch_embeddings_B.append(torch.zeros(embedding_dim, device=device))
                continue
            if sae_model:
                # Use SAE embeddings
                sae_vector_TD = sae_model.get_acts(core_activations_TH)
                embedding = torch.mean(sae_vector_TD, dim=0) # in this case embedding is D-dimensional
            else:
                # Use raw ESM embeddings
                embedding = torch.mean(core_activations_TH, dim=0) # embedding is H-dimensional
            batch_embeddings_B.append(embedding)
        embeddings_N.extend([emb.cpu().numpy() for emb in batch_embeddings_B])
    return np.array(embeddings_N)

def get_logits_embeddings(seqs, esm_model, tokenizer, batch_size=64, embedding_dim=None):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
