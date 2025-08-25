import os
import re
import sys
import time # Added for timing
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
from tqdm import tqdm

sys.path.append('..') 
sys.path.append('scoring_models')
from scoring_utils import score_mlp, score_probe_model, load_probe_model
from gen_utils import get_mutations_str, load_sae_model
from steering_utils import steer_sequence_latent
from esm.data import Alphabet

# --- Argument Parser ---
def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Perform latent steering on protein sequences using an SAE.")
    
    # Required Paths
    parser.add_argument("--dms_id", type=str, required=True, help="Identifier for the DMS experiment (e.g., 'GFP_AEQVI_Sarkisyan_2016').")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the trained SAE model checkpoint.")
    parser.add_argument("--probe_model_path", type=str, required=True, help="Path to the probe model (.pkl).")
    parser.add_argument("--probe_weights_csv", type=str, required=True, help="Path to the CSV containing latent weights from the probe.")
    parser.add_argument("--mlp_model_path", type=str, required=True, help="Path to the trained MLP scoring model (.pt).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV results.")

    # Optional Paths
    parser.add_argument("--ft_path", type=str, default=None, help="Optional path to a LoRA adapter for the ESM model.")
    parser.add_argument("--dms_dir", type=str, default="../DMS", help="Directory containing DMS data and metadata.")
    parser.add_argument("--esm_base_path", type=str, default="../models/esm2_t33_650M_UR50D", help="Path to the base ESM model directory.")

    # Hyperparameters
    parser.add_argument("--plm_dim", type=int, default=1280, help="Dimension of the Protein Language Model (PLM).")
    parser.add_argument("--sae_dim", type=int, default=4096, help="Dimension of the Sparse Autoencoder (SAE).")
    parser.add_argument("--layer", type=int, default=24, help="Layer from which to extract ESM activations.")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of unique sequences to generate.")
    parser.add_argument("--mutation_radius", type=int, default=5, help="Max number of residues to mutate.")

    # Steering Control
    parser.add_argument("--min_steer_val", type=float, default=-3, help="Minimum value for magnifying.")
    parser.add_argument("--max_steer_val", type=float, default=3, help="Maximum value for magnifying.")
    parser.add_argument("--stepsize_steer_val", type=float, default=0.2, help="Number of magnifying steps.")

    return parser

def main(args):
    """
    Main function to run the latent steering and scoring workflow.
    """
    torch.random.manual_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 1. Load Metadata and WT Sequence ---
    meta_csv_path = os.path.join(args.dms_dir, "DMS_substitutions.csv")
    df_meta = pd.read_csv(meta_csv_path)
    
    dms_row = df_meta[df_meta["DMS_id"] == args.dms_id]
    if dms_row.empty:
        print(f"Error: DMS_id '{args.dms_id}' not found in metadata.")
        return
        
    wt_sequence = dms_row["target_seq"].values[0].upper()
    positions_path = args.mlp_model_path.replace('.pt', '.npy')
    valid_positions_1_indexed = np.load(positions_path)
    valid_positions_0_indexed = set(valid_positions_1_indexed - 1)

    # --- 2. Load Models ---
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_base_path)
    base_model = AutoModelForMaskedLM.from_pretrained(args.esm_base_path)
    
    if args.ft_path:
        print(f"Applying LoRA adapter from: {args.ft_path}")
        esm_model = PeftModel.from_pretrained(base_model, args.ft_path).to(DEVICE)
    else:
        esm_model = base_model.to(DEVICE)

    sae_model = load_sae_model(args.sae_path, args.plm_dim, args.sae_dim, DEVICE)
    probe_model = load_probe_model(args.probe_model_path)
    mlp_model = torch.load(args.mlp_model_path, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
    
    alphabet = Alphabet.from_architecture("ESM-1b")
    print("Models loaded successfully.")

    # --- 3. Prepare for Steering ---
    latents_df = pd.read_csv(args.probe_weights_csv)
    latents_df["abs_weight"] = latents_df["Weight"].abs()
    # Sort all available latents to iterate through them in order of importance
    sorted_latents = latents_df.sort_values(by="abs_weight", ascending=False)

    print(f"Steering over ({args.min_steer_val} to {args.max_steer_val}, step size {args.stepsize_steer_val})")
    multipliers = np.arange(args.min_steer_val, args.max_steer_val, args.stepsize_steer_val)

    results = []
    seen_sequences = {wt_sequence}
    
    # --- 4. Main Steering and Scoring Loop ---
    print(f"Attempting to generate {args.num_sequences} unique sequences for {args.dms_id}...")
    start_run_time = time.time() # Start timer

    with tqdm(total=args.num_sequences, desc="Unique Sequences Generated") as pbar:
        # Loop through all latents until the desired number of sequences is generated
        for _, row_lat in sorted_latents.iterrows():
            if len(results) >= args.num_sequences:
                break # Stop if we have enough sequences

            latent_dim = int(row_lat["Index"])
            candidate_results = []

            for mult in multipliers:
                current_multiplier = mult
                steered_seq = steer_sequence_latent(
                    sequence=wt_sequence, tokenizer=tokenizer, esm_model=esm_model, sae_model=sae_model,
                    alphabet=alphabet, layer=args.layer, latent_dim=latent_dim, multiplier=current_multiplier, device=DEVICE
                )

                if len(steered_seq) != len(wt_sequence): continue

                steered_seq_list = list(steered_seq)
                # Revert any mutations outside the set of allowed positions
                for i_pos in range(len(steered_seq_list)):
                    if i_pos not in valid_positions_0_indexed:
                        steered_seq_list[i_pos] = wt_sequence[i_pos]
                final_seq = ''.join(steered_seq_list)
                
                if sum(1 for a, b in zip(final_seq, wt_sequence) if a != b) > args.mutation_radius: continue

                probe_score = score_probe_model(
                    [final_seq], probe_model, tokenizer, esm_model, sae_model,
                    batch_size=64, layer=args.layer, embedding_dim=args.sae_dim
                )[0]
                candidate_results.append((probe_score, final_seq, current_multiplier))

            if not candidate_results: continue
            
            candidate_results.sort(reverse=True, key=lambda x: x[0])
            
            # --- 5. Select, Score, and Record Sequence ---
            best_score_first, seq_first, mult_first = candidate_results[0]
            mutant_str_first = get_mutations_str(wt_sequence, seq_first)
            mlp_score_first = score_mlp([seq_first], mlp_model)

            unique_seq = None
            for _, seq, mult in candidate_results:
                if seq not in seen_sequences:
                    unique_seq, unique_mult = seq, mult
                    break
            
            if unique_seq is None: continue
            
            mutant_str_unique = get_mutations_str(wt_sequence, unique_seq)
            mlp_score_unique = score_mlp([unique_seq], mlp_model)
            seen_sequences.add(unique_seq)

            results.append({
                "latent": latent_dim, "mutant": mutant_str_unique, "mutated_sequence": unique_seq,
                "multiplier": unique_mult, "MLP_score": mlp_score_unique,
                "mutant_first": mutant_str_first, "mutated_seq_first": seq_first,
                "MLP_score_first": mlp_score_first, "multiplier_first": mult_first,
            })
            pbar.update(1) # Update progress bar only when a unique sequence is added

    # --- 6. Save Final Results and Timing ---
    end_run_time = time.time()
    num_generated = len(results)

    if num_generated > 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_file, index=False)
        print(f"\nSteering complete. Saved {num_generated} results to {args.output_file}")
        
        # Calculate and save average time per sequence
        total_duration = end_run_time - start_run_time
        time_per_sequence = total_duration / num_generated
        
        base_name, _ = os.path.splitext(args.output_file)
        time_file_path = f"{base_name}_time_per_sequence.npy"
        np.save(time_file_path, np.array(time_per_sequence))
        print(f"Saved average time per sequence ({time_per_sequence:.4f}s) to {time_file_path}")

    else:
        print("\nSteering complete. No valid sequences were generated.")

    if num_generated < args.num_sequences:
        print(f"Warning: Only generated {num_generated} of the {args.num_sequences} requested sequences before running out of latents.")

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)