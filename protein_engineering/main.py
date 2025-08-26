import os
import subprocess
import glob
import pandas as pd
import numpy as np

# --- 1. Global Setup ---

# Define base directories
DMS_DIR = os.path.join("..", "DMS")
MODELS_DIR = os.path.join("..", "models")
BASE_ESM_MODEL_DIR = os.path.join(MODELS_DIR, "esm2_t33_650M_UR50D")
RESULTS_BASE_DIR = "results"

# Define experiment parameters
NUM_SEQUENCES = 50
DEFAULT_MAX_MUTATIONS = [5]
LOW_N = 24 # For probe model paths
LAYER = 24
SAE_DIM = 4096
K = 128
AUXK = 256

def run_command(command, dms_id, method_name, log_file_path):
    """
    Executes a command using subprocess and redirects its output to a log file.
    """
    print(f"Executing: {' '.join(command)}")
    print(f"  - Logging to: {os.path.basename(log_file_path)}")
    try:
        with open(log_file_path, 'w') as log_file:
            subprocess.run(
                command, 
                check=True, 
                stdout=log_file, 
                stderr=log_file,
                text=True
            )
        print(f"--- Successfully finished {method_name} for {dms_id}. ---")
    except subprocess.CalledProcessError as e:
        print(f"--- 🚨 Error running {method_name} for {dms_id}: ---")
        print(f"  - Command failed with exit code {e.returncode}.")
        print(f"  - See details in the log file: {log_file_path}")
    except FileNotFoundError:
        print(f"--- 🚨 Error: 'python' command not found. Ensure Python is in your system's PATH. ---")
    except Exception as e:
        print(f"--- 🚨 An unexpected error occurred: {e} ---")


def main():
    """
    Main orchestrator script to run all sequence generation methods
    for each DMS experiment.
    """
    # --- 2. Loop Through All DMS Datasets ---
    dms_substitution_path = os.path.join(DMS_DIR, "DMS_substitutions.csv")
    if not os.path.exists(dms_substitution_path):
        print(f"Error: DMS_substitutions.csv not found at {dms_substitution_path}.")
        return

    dms_substitutions_df = pd.read_csv(dms_substitution_path)
    csv_files = glob.glob(os.path.join(DMS_DIR, "*.csv"))
    target_csv_files = [f for f in csv_files if os.path.basename(f) != "DMS_substitutions.csv"]
    
    for csv_file_path in target_csv_files:
        csv_filename = os.path.basename(csv_file_path)
        matching_row = dms_substitutions_df[dms_substitutions_df['DMS_filename'] == csv_filename]

        if matching_row.empty:
            print(f"No match for '{csv_filename}' in DMS_substitutions.csv. Skipping.")
            continue

        dms_id = matching_row['DMS_id'].iloc[0]
        
        # Define and create the results directory
        results_dir = os.path.join(RESULTS_BASE_DIR, dms_id)
        os.makedirs(results_dir, exist_ok=True)

        # Define common model paths
        ft_path = os.path.join(MODELS_DIR, dms_id, "esm_ft")
        mlp_model_path = os.path.join("scoring_models", f"{dms_id}.pt")

        if dms_id == "SPG1_STRSG_Wu_2016":
            max_mutations_to_run = [4]
            print(f"\n>>> Applying special case for {dms_id}: Using MAX_MUTATIONS = {max_mutations_to_run}")
        else:
            max_mutations_to_run = DEFAULT_MAX_MUTATIONS

        # --- 3. Loop Through Mutation Radii ---
        for max_mut in max_mutations_to_run:
            print(f"\n{'='*20} Processing {dms_id} with mutation_radius={max_mut} {'='*20}")
            output_file = os.path.join(results_dir, f"random_{max_mut}.csv")
            if os.path.exists(output_file):
                print(f"Output exists for random (max_mutations={max_mut}). Skipping.")
            else:    
                command = [
                    "python", "random_mutants.py",
                    "--dms_id", dms_id,
                    "--mlp_model_path", mlp_model_path,
                    "--output_file", output_file,
                    "--num_sequences", str(NUM_SEQUENCES),
                    "--max_mutations", str(max_mut)
                ]
                print(f"Executing: {' '.join(command)}")
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"--- Error running random_mutants.py: {e} ---")


            # --- 4. Define Model Combinations ---
            sae_path = os.path.join(MODELS_DIR, dms_id, f"sae/l{LAYER}_dim{SAE_DIM}_k{K}_auxk{AUXK}/checkpoints/last.ckpt")
            probe_base_path = os.path.join("..", "linear_probe", "results", dms_id, "learning_curve", "probe_weights", f"seed0_rerun0_train{LOW_N}")
            
            model_combos = [
                {"name": "sae", "script": "steer_latents.py", "sae_path": sae_path, "probe_model_path": os.path.join(probe_base_path, "model_sae.pkl")},
                {"name": "esm_ft", "script": "sa_esm.py", "probe_model_path": os.path.join(probe_base_path, "model_ft.pkl")},
                {"name": "esm_ft_logits", "script": "sa_esm.py", "probe_model_path": os.path.join(probe_base_path, "model_ft_logits.pkl")}
            ]

            # --- 5. Run Latent Steering & Find Max Time ---
            print("\n--- Running Latent Steering to Determine SA Timesteps ---")
            timing_files_to_check = []
            for combo in model_combos:
                if 'sae' in combo['name']:
                    probe_weights_csv = combo['probe_model_path'].replace('.pkl', '.csv')
                    base_args = ["python", "steer_latents.py", "--dms_id", dms_id, "--mlp_model_path", mlp_model_path, "--ft_path", ft_path, "--num_sequences", str(NUM_SEQUENCES), "--layer", str(LAYER), "--mutation_radius", str(max_mut), "--sae_path", combo['sae_path'], "--probe_model_path", combo['probe_model_path'], "--probe_weights_csv", probe_weights_csv]
                    
                    output_magnify = os.path.join(results_dir, f"{combo['name']}_steer_magnify_{max_mut}.csv")
                    log_magnify = output_magnify.replace('.csv', '.log')
                    timing_files_to_check.append(output_magnify.replace('.csv', '_time_per_sequence.npy'))
                    if not os.path.exists(output_magnify):
                        run_command(base_args + ["--output_file", output_magnify], dms_id, f"Steering (Magnify) for {combo['name']}", log_magnify)

            max_time_path = None
            if timing_files_to_check:
                max_time = 0
                for tf_path in timing_files_to_check:
                    if os.path.exists(tf_path):
                        current_time = np.load(tf_path).item()
                        if current_time > max_time:
                            max_time = current_time
                            max_time_path = tf_path
                if max_time_path:
                    print(f"\nDetermined max steering time ({max_time:.4f}s) from: {os.path.basename(max_time_path)}")
                else:
                    print("\nCould not find any steering time files. SA will use default timesteps.")
            
            # --- 6. Run Simulated Annealing for All Combos ---
            print("\n--- Running Simulated Annealing Methods ---")
            for combo in model_combos:
                output_file = os.path.join(results_dir, f"{combo['name']}_sa_{max_mut}.csv")
                log_file = output_file.replace('.csv', '.log')


                command = ["python", combo['script'], "--dms_id", dms_id, "--mlp_model_path", mlp_model_path, "--ft_path", ft_path, "--output_file", output_file, "--num_sequences", str(NUM_SEQUENCES), "--mutation_radius", str(max_mut)]
                if max_time_path:
                    command.extend(["--timing_file_path", max_time_path])

                if combo['script'] == 'sa_esm.py' and 'esm_ft_logits' in output_file:
                    command.extend(["--probe_model_path", combo['probe_model_path'], "--logits"])
                elif combo['script'] == 'sa_esm.py':
                    command.extend(["--probe_model_path", combo['probe_model_path']])
                
                run_command(command, dms_id, f"SA for {combo['name']}", log_file)

if __name__ == "__main__":
    main()
