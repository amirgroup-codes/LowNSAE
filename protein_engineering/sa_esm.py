import os
import re
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
from tqdm import tqdm

# Add project paths to import custom modules
sys.path.append('..')
sys.path.append('scoring_models')
from gen_utils import apply_mutations
from scoring_utils import score_mlp, load_probe_model, score_probe_model
from esm.data import Alphabet
from sa_sampler import SimulatedAnnealing

# --- ESM Scoring Helpers ---
def label_row(row, wt_seq, token_probs, alphabet):
    """Calculates log-odds score for a single mutated sequence."""
    mut_list = np.array(list(row))
    wt_list = np.array(list(wt_seq))
    idx = np.where(mut_list != wt_list)[0]
    if len(idx) == 0:
        return 0.0
    mut_values = mut_list[idx]
    wt_values = wt_list[idx]
    scores = []
    for wt, mut, ind in zip(wt_values, mut_values, idx):
        wt_encoded = alphabet.get_idx(wt)
        mt_encoded = alphabet.get_idx(mut)
        score = token_probs[0, 1 + ind, mt_encoded] - token_probs[0, 1 + ind, wt_encoded]
        scores.append(score.item())
    return sum(scores)

def score_sequence_esm(mut_seq: str, wt_seq: str, esm_model, alphabet, batch_converter, device='cpu') -> float:
    """Gets the ESM log-odds score for a single sequence by using the model's final logits."""
    esm_model.eval()
    data = [("wt", wt_seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        token_probs = torch.log_softmax(esm_model(batch_tokens)["logits"], dim=-1).cpu()
    score = label_row(mut_seq, wt_seq, token_probs, alphabet)
    return score

def process_mutant_string(mutant_str):
    """
    Parses a mutant string, adds 1 to each position, and sorts the mutations numerically.
    
    Args:
        mutant_str (str): A string like 'V279P:D265W:V264G:G266R'.
        
    Returns:
        str: The processed and sorted string, e.g., 'V265G:D266W:G267R:V280P'.
    """
    mutations = mutant_str.split(':')
    parsed_mutations = []
    
    for mut in mutations:
        # Use regex to capture the parts: original AA, position, and new AA
        match = re.match(r"([A-Z])(\d+)([A-Z])", mut)
        if match:
            original_aa, pos, new_aa = match.groups()
            parsed_mutations.append({
                'original_aa': original_aa,
                'position': int(pos),
                'new_aa': new_aa
            })
            
    # Sort the list of dictionaries based on the original position
    parsed_mutations.sort(key=lambda x: x['position'])
    
    # Create the new mutant strings with the position incremented by 1
    new_mutant_parts = []
    for mut_info in parsed_mutations:
        new_position = mut_info['position'] + 1
        new_mut_str = f"{mut_info['original_aa']}{new_position}{mut_info['new_aa']}"
        new_mutant_parts.append(new_mut_str)
        
    return ":".join(new_mutant_parts)

# --- Argument Parser ---
def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Perform Simulated Annealing using ESM scores to generate optimized protein sequences.")
    
    # Required Paths
    parser.add_argument("--dms_id", type=str, required=True, help="Identifier for the DMS experiment.")
    parser.add_argument("--mlp_model_path", type=str, required=True, help="Path to the trained MLP scoring model (.pt).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV results.")

    # Optional Paths for Fitness Function
    parser.add_argument("--probe_model_path", type=str, default=None, help="Path to a probe model (.pkl). If provided, SA uses ESM+Probe. Otherwise, uses ESM log-odds.")
    parser.add_argument("--ft_path", type=str, default=None, help="Optional path to a LoRA adapter for the ESM model.")
    parser.add_argument("--dms_dir", type=str, default="../DMS", help="Directory containing DMS data.")
    parser.add_argument("--esm_base_path", type=str, default="../models/esm2_t33_650M_UR50D", help="Path to the base ESM model.")
    parser.add_argument("--timing_file_path", type=str, default=None, help="Path to .npy file with time per sequence from another method. Used to calibrate SA timesteps.")

    # Hyperparameters
    parser.add_argument("--plm_dim", type=int, default=1280, help="Dimension of the Protein Language Model (PLM).")
    parser.add_argument("--layer", type=int, default=24, help="Layer from which to extract ESM activations for the probe. For log-odds, the final layer is always used.")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of unique sequences to generate.")
    parser.add_argument("--mutation_radius", type=int, default=5, help="Number of mutations to introduce in each SA step.")
    parser.add_argument("--logits", action='store_true', default=False, help="Use log-odds instead of logits for the probe.")
    
    # Simulated Annealing Control
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps for SA. Overridden by timing calibration.")
    parser.add_argument("--t_init", type=float, default=1e2, help="Initial temperature for SA.")
    parser.add_argument("--t_final", type=float, default=1e-2, help="Final temperature for SA.")
    parser.add_argument("--mutation_rate", type=float, default=3, help="Mutation rate for SA.")

    return parser

def main(args):
    """Main function to run the simulated annealing and scoring workflow."""
    torch.random.manual_seed(42)
    np.random.seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

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
    valid_positions_0_indexed = valid_positions_1_indexed - 1

    # --- 2. Load Models ---
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_base_path)
    base_model = AutoModelForMaskedLM.from_pretrained(args.esm_base_path)
    
    if args.ft_path:
        print(f"Applying LoRA adapter from: {args.ft_path}")
        model = PeftModel.from_pretrained(base_model, args.ft_path).to(DEVICE)
    else:
        model = base_model.to(DEVICE)

    mlp_model = torch.load(args.mlp_model_path, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
    alphabet = Alphabet.from_architecture("ESM-1b")
    batch_converter = alphabet.get_batch_converter()
    print("Models loaded successfully.")

    # --- 3. Setup Fitness Function and Sampler ---
    fitness_col_name = ""
    if args.probe_model_path and args.logits is False:
        # --- Mode 1: ESM + Linear Probe ---
        print(f"Mode: Simulated Annealing with ESM + Linear Probe on layer {args.layer}.")
        fitness_col_name = "LinearProbe_score"
        probe_model = load_probe_model(args.probe_model_path)

        def seq2fitness_probe_esm(mutant_str, wt_sequence_arg):
            try:
                seq = apply_mutations(wt_sequence_arg, mutant_str)
            except Exception:
                raise ValueError(f"Invalid mutant string: {mutant_str}")
            return score_probe_model([seq], probe_model, tokenizer, esm_model=model, sae_model=None,
                layer=args.layer, embedding_dim=args.plm_dim)[0]
        
        seq2fitness = seq2fitness_probe_esm
    elif args.probe_model_path and args.logits:
        # --- Mode 2: ESM + Linear Probe (logits) ---
        print(f"Mode: Simulated Annealing with ESM + Linear Probe on logits.")
        fitness_col_name = "LinearProbe_score"
        probe_model = load_probe_model(args.probe_model_path)

        def seq2fitness_probe_esm(mutant_str, wt_sequence_arg):
            try:
                seq = apply_mutations(wt_sequence_arg, mutant_str)
            except Exception:
                raise ValueError(f"Invalid mutant string: {mutant_str}")
            return score_probe_model([seq], probe_model, tokenizer, esm_model=model, sae_model=None,
                layer=args.layer, embedding_dim=args.plm_dim, logits=True)[0]
        
        seq2fitness = seq2fitness_probe_esm
    else:
        # --- Mode 3: ESM Log-Odds Score Only ---
        print("Mode: Simulated Annealing with ESM log-odds score (from final layer logits).")
        fitness_col_name = "ESM_score"

        def seq2fitness_esm(mutant_str, wt_sequence_arg):
            try:
                seq = apply_mutations(wt_sequence_arg, mutant_str)
                return score_sequence_esm(seq, wt_sequence_arg, model, alphabet, batch_converter, device=DEVICE)
            except Exception as e:
                raise ValueError(f"Error scoring sequence {mutant_str}")
        
        seq2fitness = seq2fitness_esm

    # Vary number of mutations by a poisson distribution
    mutation_rate_radius = round(args.mutation_radius / 2)

    # --- 4. Determine Timesteps for SA ---
    timesteps = None
    base_name, _ = os.path.splitext(args.output_file)
    timestep_save_path = f"{base_name}_num_timesteps_esm.npy"

    if os.path.exists(timestep_save_path):
        timesteps = int(np.load(timestep_save_path).item())
        print(f"Loaded pre-calculated timesteps: {timesteps}")
    elif args.timing_file_path and os.path.exists(args.timing_file_path):
        print("Calibrating SA timesteps based on provided timing file...")
        target_time_per_seq = np.load(args.timing_file_path).item()
        calibration_steps = 1000
        sampler = SimulatedAnnealing(
            seq2fitness=seq2fitness,
            AA_options=[AA_ALPHABET if i in valid_positions_0_indexed else [aa] for i, aa in enumerate(wt_sequence)],
            WT=wt_sequence,
            seed=42,
            split_char=":",
            mutation_rate=args.mutation_rate,
            number_mutations=args.mutation_radius
        )
        print(f"Running calibration with {calibration_steps} steps...")
        start_cal_time = time.time()
        sampler.walk(np.logspace(np.log10(args.t_init), np.log10(args.t_final), calibration_steps))
        end_cal_time = time.time()

        time_per_timestep = (end_cal_time - start_cal_time) / calibration_steps
        timesteps = int(target_time_per_seq / time_per_timestep)
        
        np.save(timestep_save_path, np.array(timesteps))
        print(f"Calibration complete. Target time: {target_time_per_seq:.4f}s.")
        print(f"Time per SA step: {time_per_timestep:.6f}s.")
        print(f"Calculated timesteps for SA: {timesteps}. Saved to {timestep_save_path}")
    else:
        timesteps = args.timesteps
        print(f"Using default or user-specified timesteps: {timesteps}")


    # --- 5. Main SA and Scoring Loop ---
    print(f"\nStarting {args.num_sequences} simulated annealing runs of {timesteps} steps each...")
    mutation_schedule = np.logspace(np.log10(args.t_init), np.log10(args.t_final), timesteps)
    results = []
    seen_sequences = {wt_sequence}
    skip_seen_check = timesteps > 1000 # Skip checking if timesteps > 1000
    with tqdm(total=args.num_sequences, desc="Simulated Annealing ESM runs") as pbar:
        seed = 0
        while len(results) < args.num_sequences:

            # Sample from mutation_rate_radius to determine the number of mutations to make
            m = np.random.poisson(mutation_rate_radius) + 1
            if m > args.mutation_radius:
                m = args.mutation_radius

            sampler = SimulatedAnnealing(
                seq2fitness=seq2fitness,
                AA_options=[AA_ALPHABET if i in valid_positions_0_indexed else [aa] for i, aa in enumerate(wt_sequence)],
                WT=wt_sequence,
                seed=seed,
                split_char=":",
                mutation_rate=args.mutation_rate,
                number_mutations=m
            )

            df_walk = sampler.walk(mutation_schedule)
            
            best_mutant_str = df_walk.iloc[-1].best_mutant
            best_seq = apply_mutations(wt_sequence, best_mutant_str)
            seed += 1

            # Check for uniqueness only if we are not skipping the check.
            # If skip_seen_check is True, the condition is always met.
            if skip_seen_check or best_seq not in seen_sequences:
                seen_sequences.add(best_seq)
                
                fitness_score = df_walk.iloc[-1].best_fitness
                mlp_score = score_mlp([best_seq], mlp_model)
                
                result_entry = {
                    "run": len(results),
                    "mutant": best_mutant_str,
                    "mutated_sequence": best_seq,
                    "MLP_score": mlp_score,
                }
                result_entry[fitness_col_name] = fitness_score
                results.append(result_entry)
                
                pbar.update(1)
            
    # --- 6. Save Final Results ---
    if results:
        results_df = pd.DataFrame(results)
        cols = ["run", "mutant", "mutated_sequence", fitness_col_name, "MLP_score"]
        results_df = results_df[cols]
        results_df['mutant'] = results_df['mutant'].apply(process_mutant_string)
        results_df.to_csv(args.output_file, index=False)
        print(f"\nSimulated annealing complete. Saved {len(results_df)} results to {args.output_file}")
    else:
        print("\nSimulated annealing finished, but no results were generated.")

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)