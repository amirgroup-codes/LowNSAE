import torch
import numpy as np
import pandas as pd
from interprot.interprot.utils import get_layer_activations

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

def steer_sequence_latent(
    sequence,
    tokenizer,
    esm_model,
    sae_model,
    alphabet,
    layer,
    latent_dim,
    multiplier=1.0,
    similarity_threshold=0.98,
    device="cuda"
):
    """
    Steers a sequence and keeps mutations only if the final output logits differ
    significantly from the wildtype logits, based on cosine similarity.

    Args:
        sequence (str): The input sequence to be steered (acts as both target and wildtype).
        tokenizer: HuggingFace tokenizer for ESM.
        esm_model: ESM model, possibly with a LoRA adapter.
        sae_model: Trained SparseAutoencoder.
        alphabet: ESM alphabet object.
        layer (int): Layer from which to extract activations for steering.
        latent_dim (int): Index of the SAE latent dimension to steer.
        multiplier (float): The factor by which to steer the latent activation.
        similarity_threshold (float): Cosine similarity threshold (e.g., 0.99). If similarity is
                                      above this, the original amino acid is kept.
        device (str or torch.device): The device to run computations on.

    Returns:
        str: The final steered amino acid sequence with conditional mutations.
    """
    device = torch.device(device)
    esm_model.eval()
    sae_model.eval()

    # --- Pass 1: Generate Steered Logits ---
    # Get ESM representations up to the steering layer
    esm_layer_acts_TH = get_layer_activations(
        tokenizer, esm_model, [sequence], layer=layer
    )[0][1:-1].to(device)

    # Encode through SAE and calculate reconstruction error
    with torch.no_grad():
        sae_latents_TD, mu, std = sae_model.encode(esm_layer_acts_TH)
        reconstructed_acts_TH = sae_model.decode(sae_latents_TD, mu, std)
    reconstruction_error_TH = esm_layer_acts_TH - reconstructed_acts_TH

    # Modify the target latent dimension
    base_activation_T = sae_latents_TD.max() if multiplier > 0 else sae_latents_TD.min()
    sae_latents_TD[:, latent_dim] = base_activation_T * multiplier

    # Decode from steered latents and pass through remaining ESM layers
    with torch.no_grad():
        steered_acts_TH = sae_model.decode(sae_latents_TD, mu, std) + reconstruction_error_TH
        hidden_states_1TH = steered_acts_TH.unsqueeze(0)
        attention_mask_1T = torch.ones((1, hidden_states_1TH.shape[1]), dtype=torch.long, device=device)
        for layer_module in esm_model.esm.encoder.layer[layer:]:
            hidden_states_1TH = layer_module(hidden_states_1TH, attention_mask=attention_mask_1T)[0]
        steered_logits_1TV = esm_model.lm_head(hidden_states_1TH)

    # --- Pass 2: Generate Wildtype Logits ---
    wt_inputs = tokenizer(sequence, return_tensors="pt").to(device)
    with torch.no_grad():
        wt_outputs = esm_model(**wt_inputs)
        # Slice to remove <cls> and <eos> tokens for comparison
        wt_logits_1TV = wt_outputs.logits[:, 1:-1, :]

    # --- 6. Compare Logits and Decode Final Sequence ---
    
    # Calculate the cosine similarity between the logit vectors at each position
    cos = torch.nn.CosineSimilarity(dim=-1)
    # The output 'similarities' tensor will have shape [1, sequence_length]
    similarities_1T = cos(steered_logits_1TV, wt_logits_1TV)

    # Get the original token IDs for the wildtype sequence
    original_ids = wt_inputs['input_ids'][:, 1:-1]
    
    # Get the new predicted token IDs from the steered logits
    steered_token_ids_1T = torch.argmax(steered_logits_1TV, dim=-1)

    # Create a mask where the logits are considered too similar to warrant a mutation
    is_similar_mask_1T = (similarities_1T >= similarity_threshold)

    # Where the mask is True (logits are similar), use the original token.
    # Where the mask is False (logits are different), use the new steered token.
    final_token_ids_1T = torch.where(
        is_similar_mask_1T,
        original_ids,
        steered_token_ids_1T
    )

    # Convert final token IDs to an amino acid sequence
    tokens_T = [alphabet.get_tok(t.item()) for t in final_token_ids_1T[0]]
    aa_tokens_T = [tok for tok in tokens_T if len(tok) == 1 and tok.isalpha()]
    
    return ''.join(aa_tokens_T)