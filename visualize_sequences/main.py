import pandas as pd
import numpy as np
import os
import torch
import sys
sys.path.append('..')
from interprot.interprot.utils import get_layer_activations
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
from gen_utils import get_mutations_str, load_sae_model
from pathlib import Path
from visualize_utils import extract_top_peaks, analyze_activation_differences, format_peaks_for_output, generate_visualization_files
import re
import shutil

DMS_DIR = os.path.join("..", "DMS")
MODELS_DIR = os.path.join("..", "models")
BASE_ESM_MODEL_DIR = os.path.join(MODELS_DIR, "esm2_t33_650M_UR50D")
PDB_DIR = "pdbs"
RESULTS_DIR = "results"
LAYER = 24
PLM_DIM = 1280
SAE_DIM = 4096
K = 128
AUXK = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = 24


def main():
    """
    Main orchestrator script to run all sequence generation methods
    for each DMS experiment.
    """
    dms_substitution_path = os.path.join(DMS_DIR, "DMS_substitutions.csv")
    if not os.path.exists(dms_substitution_path):
        print(f"Error: DMS_substitutions.csv not found at {dms_substitution_path}.")
        return
    df_meta = pd.read_csv(dms_substitution_path)
    pdb_folders = [f for f in os.listdir(PDB_DIR) if os.path.isdir(os.path.join(PDB_DIR, f))]

    for pdb_folder in pdb_folders:
        print(f"Processing {pdb_folder}")

        dms_row = df_meta[df_meta["DMS_id"] == pdb_folder]
        if dms_row.empty:
            print(f"Error: DMS_id '{pdb_folder}' not found in metadata.")
            return
        wt_sequence = dms_row["target_seq"].values[0].upper()
        dms_id = dms_row["DMS_id"].values[0]

        # --- 2. Create Output Directory ---
        output_dir = Path(RESULTS_DIR) / pdb_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_txt_path = output_dir / f"highlighted_latents_layer{LAYER}.txt"

         # --- 2. Load Models ---
        sae_msa_path = os.path.join(MODELS_DIR, dms_id, f"sae_ft_msa/l{LAYER}_dim{SAE_DIM}_k{K}_auxk{AUXK}/checkpoints/last.ckpt")
        ft_path = os.path.join(MODELS_DIR, dms_id, "esm_ft")
        print("Loading models...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_ESM_MODEL_DIR)
        base_model = AutoModelForMaskedLM.from_pretrained(BASE_ESM_MODEL_DIR)
        esm_model = PeftModel.from_pretrained(base_model, ft_path).to(DEVICE)
        sae_model = load_sae_model(sae_msa_path, PLM_DIM, SAE_DIM, DEVICE)
        
        csv_path = Path(f"../linear_probe/results/{pdb_folder}/learning_curve/probe_weights/seed0_rerun0_train{N_TRAIN}/model_ft_msa.csv")
        if not csv_path.exists():
            print(f"Warning: Probe weights CSV not found at {csv_path}. Skipping analysis for {pdb_folder}.")
            continue
        weights_df = pd.read_csv(csv_path)

        # --- 4. Analyze Wildtype Sequence ---
        print("Analyzing Wildtype Sequence...")
        with torch.no_grad():
            layer_output_wt = get_layer_activations(tokenizer, esm_model, [wt_sequence], layer=LAYER)[0][1:-1].to(DEVICE)
            sae_activations_wt = sae_model.get_acts(layer_output_wt).cpu().numpy()
        wt_peaks = extract_top_peaks(weights_df, sae_activations_wt)
        wt_output_lines = format_peaks_for_output(wt_peaks, wt_sequence)

        # --- 5. Load and Analyze Top Mutant Sequence ---
        print("Analyzing Top Mutant Sequence...")
        mutant_seq_path_5 = Path(f"../protein_engineering/results/{pdb_folder}/sae_ft_msa_steer_magnify_5.csv")
        mutant_seq_path_4 = Path(f"../protein_engineering/results/{pdb_folder}/sae_ft_msa_steer_magnify_4.csv")
        
        mutant_seq_path = None
        if mutant_seq_path_5.exists():
            mutant_seq_path = mutant_seq_path_5
        elif mutant_seq_path_4.exists():
            mutant_seq_path = mutant_seq_path_4
            
        mut_peaks = {}
        mut_output_lines = {"Positive": ["No mutant data found."], "Negative": ["No mutant data found."]}
        top_mutant_id = "N/A"

        if mutant_seq_path:
            df_mut = pd.read_csv(mutant_seq_path)
            
            # Determine score column
            if pdb_folder == "SPG1_STRSG_Wu_2016":
                dms_df_path = os.path.join(DMS_DIR, f"{pdb_folder}.csv")
                if os.path.exists(dms_df_path):
                    dms_df = pd.read_csv(dms_df_path)
                    merged_df = pd.merge(df_mut, dms_df[['mutant', 'DMS_score']], on='mutant', how='left')
                    score_column = "DMS_score"
                else:
                    print(f"Warning: DMS score file not found for {pdb_folder}. Defaulting to MLP_score.")
                    merged_df = df_mut
                    score_column = "MLP_score"
            else:
                merged_df = df_mut
                score_column = "MLP_score"

            if score_column not in merged_df.columns:
                print(f"Warning: Score column '{score_column}' not found in {mutant_seq_path}. Skipping.")
                continue
                
            merged_df = merged_df.sort_values(by=score_column, ascending=False).dropna(subset=[score_column])
            if merged_df.empty:
                print(f"No valid mutants with scores found in {mutant_seq_path}. Skipping.")
                continue
                
            # --- Loop through top mutants (REVISED SECTION) ---
            print(f"Analyzing top 5 mutants from {mutant_seq_path}...")
            for _, top_mutant in merged_df.head(5).iterrows():
                mut_sequence = top_mutant["mutated_sequence"].upper()
                top_mutant_id = top_mutant["mutant"]
                
                # 1. Sanitize ID and create the dedicated output directory
                af_id = top_mutant_id.lower().replace(":", "_").replace("/", "_")
                mutant_output_dir = Path(RESULTS_DIR) / pdb_folder / af_id
                mutant_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 2. Find, copy, and rename the CIF file
                source_cif_path = Path(PDB_DIR) / pdb_folder / f"fold_{af_id}_model_0.cif"
                dest_cif_path = mutant_output_dir / f"{af_id}.cif"
                
                if not source_cif_path.exists():
                    print(f"  - WARNING: Source CIF not found for {af_id}. Skipping this mutant.")
                    continue
                
                shutil.copy(source_cif_path, dest_cif_path)

                # 3. Run analyses
                with torch.no_grad():
                    layer_output_mut = get_layer_activations(tokenizer, esm_model, [mut_sequence], layer=LAYER)[0][1:-1].to(DEVICE)
                    sae_activations_mut = sae_model.get_acts(layer_output_mut).cpu().numpy()
                
                mut_peaks = extract_top_peaks(weights_df, sae_activations_mut)
                mut_output_lines = format_peaks_for_output(mut_peaks, mut_sequence)
                diff_output_lines = analyze_activation_differences(sae_activations_wt, sae_activations_mut, mut_sequence, weights_df)
                
                # 4. Write the analysis .txt file into the new directory
                analysis_txt_path = mutant_output_dir / f"{af_id}_layer{LAYER}_analysis.txt"
                with open(analysis_txt_path, "w") as f:
                    f.write("========== Wildtype ==========\n")
                    f.write("# === Top Positive Latents (from probe) ===\n")
                    f.write("\n".join(wt_output_lines["Positive"]) + "\n")
                    f.write("\n# === Top Negative Latents (from probe) ===\n")
                    f.write("\n".join(wt_output_lines["Negative"]) + "\n")

                    f.write(f"\n======== Mutant {top_mutant_id} ========\n")
                    f.write("# === Top Positive Latents (from probe) ===\n")
                    f.write("\n".join(mut_output_lines["Positive"]) + "\n")
                    f.write("\n# === Top Negative Latents (from probe) ===\n")
                    f.write("\n".join(mut_output_lines["Negative"]) + "\n")
                    
                    f.write("\n======== Latent Difference (Mutant - Wildtype) ========\n")
                    f.write("# === Positive Latents with Changed Activation ===\n")
                    f.write("\n".join(diff_output_lines["Positive"]) + "\n")
                    f.write("\n# === Negative Latents with Changed Activation ===\n")
                    f.write("\n".join(diff_output_lines["Negative"]) + "\n")

                # 5. Generate all visualization files in the new directory
                generate_visualization_files(af_id, top_mutant_id, diff_output_lines, mutant_output_dir)

if __name__ == "__main__":
    main()