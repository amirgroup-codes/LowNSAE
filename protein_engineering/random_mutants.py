import os
import re
import sys
import random
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add project paths to import custom modules
sys.path.append('..')
sys.path.append('scoring_models')
from scoring_utils import score_mlp
from gen_utils import get_mutations_str

def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Generate and score random mutants of a protein sequence.")
    
    # Required Paths & Identifiers
    parser.add_argument("--dms_id", type=str, required=True, help="Identifier for the DMS experiment (e.g., 'GFP_AEQVI_Sarkisyan_2016').")
    parser.add_argument("--mlp_model_path", type=str, required=True, help="Path to the trained MLP scoring model (.pt).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV of random mutants.")

    # Optional Paths & Parameters
    parser.add_argument("--dms_dir", type=str, default="../DMS", help="Directory containing DMS data and metadata.")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of unique random sequences to generate.")
    parser.add_argument("--max_mutations", type=int, default=10, help="The maximum number of mutations allowed in a sequence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser

def apply_random_mutations(wt_seq, mut_count, valid_positions, aa_list):
    """Applies a specific number of random mutations to a sequence at valid positions."""
    seq = list(wt_seq)
    
    # Ensure mutation count doesn't exceed the number of available positions
    mut_count = min(mut_count, len(valid_positions))

    # Choose random positions from the provided list of valid (0-indexed) sites
    mutated_indices = np.random.choice(
        valid_positions, size=mut_count, replace=False
    )
    
    for pos in mutated_indices:
        current_aa = seq[pos]
        possible_mutants = [aa for aa in aa_list if aa != current_aa]
        seq[pos] = random.choice(possible_mutants)
    return ''.join(seq)

def main(args):
    """Main function to generate, score, and save random mutants."""
    # --- 1. Setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 2. Load Metadata and WT Sequence ---
    meta_csv_path = os.path.join(args.dms_dir, "DMS_substitutions.csv")
    df_meta = pd.read_csv(meta_csv_path)
    
    row = df_meta[df_meta["DMS_id"] == args.dms_id]
    if row.empty:
        print(f"Error: DMS_id '{args.dms_id}' not found in metadata.")
        return
        
    wt_sequence = row["target_seq"].values[0].upper()
    positions_path = args.mlp_model_path.replace('.pt', '.npy')
    print(f"Loading valid positions from: {positions_path}")
    valid_positions_1_indexed = np.load(positions_path)
    # Convert from 1-indexed to 0-indexed
    valid_positions_0_indexed = valid_positions_1_indexed - 1

    # --- 3. Load Scoring Model ---
    print(f"Loading MLP model from: {args.mlp_model_path}")
    # Load the entire model object, assuming it was saved with torch.save(model, path)
    mlp_model = torch.load(args.mlp_model_path, map_location=DEVICE, weights_only=False).eval()

    # --- 4. Generate and Score Mutants ---
    results = []
    mutation_rate = round(args.max_mutations / 2)
    print(f"Generating {args.num_sequences} random mutants...")
    for _ in tqdm(range(args.num_sequences)):
        # Sample number of mutations from a Poisson distribution
        m = np.random.poisson(mutation_rate) + 1
        if m > args.max_mutations:
            m = args.max_mutations

        mutated_seq = apply_random_mutations(wt_sequence, m, valid_positions_0_indexed, AA_LIST)
        mutant_str = get_mutations_str(wt_sequence, mutated_seq)
        mlp_score = score_mlp([mutated_seq], mlp_model)

        results.append({
            "mutant": mutant_str,
            "mutated_sequence": mutated_seq,
            "MLP_score": mlp_score
        })

    # --- 5. Save Results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)