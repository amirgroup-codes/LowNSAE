import os
import pandas as pd
import subprocess
import glob
import time
import numpy as np
import argparse
from collections import deque # For managing GPU queue
import sys
sys.path.append('..')
from finetune.main import get_num_sequences_in_a2m

# --- Global Variables ---
script_dir = os.path.dirname(__file__)
DMS_DIR = os.path.join(script_dir, "..", "DMS")
MSA_FILES_DIR = os.path.join(script_dir, "..", "MSA_files")
ESM_MODEL_DIR = os.path.join(script_dir, "..", "models", "esm2_t33_650M_UR50D")
TRAINING_SCRIPT = os.path.join(script_dir, "..", "interprot", "interprot", "training.py")
CREATE_DATA_SCRIPT = os.path.join(script_dir, "create_data.py")

# Global variable for GPUs
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, default="3,4,5", help="Comma-separated list of GPU IDs to use")
args = parser.parse_args()
GPUS = [int(gpu) for gpu in args.gpus.split(",")]

def determine_epochs_sae(num_sequences):
    """Determines the number of epochs for DMS training based on sequence count."""
    if num_sequences < 500:
        return 1000
    elif 500 <= num_sequences < 1000:
        return 500
    elif 1000 <= num_sequences < 5000:
        return 100
    else: # 5000 or more sequences
        return 10

def main():

    if not os.path.exists(CREATE_DATA_SCRIPT):
        print(f"Error: {CREATE_DATA_SCRIPT} not found in the current directory.")
        return
    if not os.path.exists(TRAINING_SCRIPT):
        print(f"Error: {TRAINING_SCRIPT} not found.")
        return
    dms_substitution_path = os.path.join(DMS_DIR, "DMS_substitutions.csv")
    if not os.path.exists(dms_substitution_path):
        print(f"Error: DMS_substitutions.csv not found at {dms_substitution_path}.")
        return
    try:
        dms_substitutions_df = pd.read_csv(dms_substitution_path)
    except Exception as e:
        print(f"Error reading DMS_substitutions.csv: {e}")
        return
    csv_files = glob.glob(os.path.join(DMS_DIR, "*.csv"))
    target_csv_files = [f for f in csv_files if os.path.basename(f) != "DMS_substitutions.csv"]
    if not target_csv_files:
        print(f"No target DMS CSV files found in {DMS_DIR}. Exiting.")
        return
    
    print("\n--- Starting SAE training on DMS-specific MSAs ---")
    running_processes = deque()
    available_gpus = deque(GPUS)
    for csv_file_path in target_csv_files:
        csv_filename = os.path.basename(csv_file_path)
        print(f"\n--- Processing DMS Experiment: {csv_filename} ---")

        matching_row = dms_substitutions_df[dms_substitutions_df['DMS_filename'] == csv_filename]

        if matching_row.empty:
            print(f"No matching entry found for '{csv_filename}' in DMS_substitutions.csv. Skipping.")
            continue

        msa_filename = matching_row['MSA_filename'].iloc[0]
        dms_id = matching_row['DMS_id'].iloc[0]

        msa_filepath = os.path.join(MSA_FILES_DIR, msa_filename)
        if not os.path.exists(msa_filepath):
            print(f"MSA file not found: {msa_filepath}. Skipping for {csv_filename}.")
            continue

        sae_ft_output_dir = os.path.join(script_dir, "..", "models", str(dms_id), "sae")
        sae_adapter_model_path = os.path.join(sae_ft_output_dir, "l24_dim4096_k128_auxk256/checkpoints", "last.ckpt")

        if os.path.exists(sae_adapter_model_path):
            print(f"  {sae_adapter_model_path} already exists. Skipping SAE training for {csv_filename}.")
            continue
        
        os.makedirs(sae_ft_output_dir, exist_ok=True)
        print(f"  SAE Fine-tune Output directory: {sae_ft_output_dir}")

        dms_parquet_output_dir = os.path.join(script_dir, "..", "training_data")
        os.makedirs(dms_parquet_output_dir, exist_ok=True)
        dms_parquet_path = os.path.join(dms_parquet_output_dir, f"{dms_id}.parquet")

        print(f"  Running create_data.py for {msa_filename} -> {dms_parquet_path}")
        create_data_command = ["python", CREATE_DATA_SCRIPT, "--msa-path", msa_filepath, "--output-parquet", dms_parquet_path]
        try:
            subprocess.run(create_data_command, check=True, capture_output=True, text=True)
            print(f"  Successfully created {dms_parquet_path}.")
        except subprocess.CalledProcessError as e:
            print(f"  Error running create_data.py for {csv_filename}: {e.returncode}\n  STDERR:\n{e.stderr}")
            continue
        except FileNotFoundError:
            print(f"  Error: '{CREATE_DATA_SCRIPT}' command not found. Skipping training for {csv_filename}.")
            continue

        num_sequences_in_msa = get_num_sequences_in_a2m(msa_filepath)
        if num_sequences_in_msa == 0:
            print(f"  No sequences found in {msa_filepath}. Skipping training for {csv_filename}.")
            continue
        num_epochs = determine_epochs_sae(num_sequences_in_msa)
        print(f"  MSA has {num_sequences_in_msa} sequences. Assigned {num_epochs} epochs for training.")

        esm_ft_adapter_path = os.path.join(script_dir, "..", "models", str(dms_id), "esm_ft", "adapter_model.safetensors")
        if not os.path.exists(esm_ft_adapter_path):
            print(f"  ESM fine-tuned adapter not found: {esm_ft_adapter_path}. Skipping training for {csv_filename}.")
            continue

        while len(running_processes) >= len(GPUS):
            print("  All GPUs are busy. Waiting for a process to complete...")
            freed_gpu = False
            while running_processes:
                process, _, _, _, _ = running_processes[0] 
                if process.poll() is not None:
                    process, gpu_id, proc_filename, log_file_path, log_file = running_processes.popleft()
                    log_file.close()
                    
                    print(f"  Process for {proc_filename} on GPU {gpu_id} finished.")
                    available_gpus.append(gpu_id)

                    if process.returncode != 0:
                        print(f"  Training for {proc_filename} on GPU {gpu_id} failed with exit code {process.returncode}.")
                        print(f"  Check log file for details: {log_file_path}")
                    else:
                        print(f"  Training for {proc_filename} on GPU {gpu_id} completed successfully.")
                    freed_gpu = True
                    break
                else:
                    time.sleep(5)
            if not freed_gpu:
                time.sleep(10)

        current_gpu = available_gpus.popleft()
        print(f"  Starting training for {csv_filename} on GPU {current_gpu}.")

        training_command_dms = [
            "python", TRAINING_SCRIPT, "--data-dir", dms_parquet_path, "--esm2-weight", esm_ft_adapter_path,
            "--d-model", "1280", "--d-hidden", "4096", "--lr", "2e-4", "--max-epochs", str(num_epochs),
            "--num-devices", "1", "--num-workers", "4", "--output_dir", sae_ft_output_dir
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(current_gpu)
        log_file_path = os.path.join(sae_ft_output_dir, "log.txt")
        print(f"  Logging output to: {log_file_path}")

        try:
            log_file = open(log_file_path, 'w')
            process = subprocess.Popen(training_command_dms, env=env, stdout=log_file, stderr=log_file, text=True)
            running_processes.append((process, current_gpu, csv_filename, log_file_path, log_file)) 
            print(f"  Started training for {csv_filename} (PID: {process.pid}) on GPU {current_gpu}.")
        except Exception as e:
            print(f"  An unexpected error occurred when launching training for {csv_filename}: {e}. Releasing GPU {current_gpu}.")
            available_gpus.append(current_gpu)
            if 'log_file' in locals() and not log_file.closed:
                log_file.close()

    print("\n--- All DMS-specific training launched. Waiting for remaining processes to complete ---")
    while running_processes:
        process, gpu_id, filename, log_file_path, log_file = running_processes.popleft()
        print(f"  Waiting for {filename} (PID: {process.pid}) on GPU {gpu_id} to finish...")
        
        process.wait() 
        log_file.close()

        if process.returncode != 0:
            print(f"  Training for {filename} on GPU {gpu_id} failed with exit code {process.returncode}.")
            print(f"  Check log file for details: {log_file_path}")
        else:
            print(f"  Training for {filename} on GPU {gpu_id} completed successfully.")
        available_gpus.append(gpu_id)

    print("\n--- All DMS-specific training processed. ---")

if __name__ == "__main__":
    main()