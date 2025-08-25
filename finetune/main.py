import os
import pandas as pd
import subprocess
import glob

DMS_DIR = os.path.join("..", "DMS")
MSA_FILES_DIR = os.path.join("..", "MSA_files")
ESM_MODEL_DIR = os.path.join("..", "models", "esm2_t33_650M_UR50D")
FINETUNE_SCRIPT = "finetune.py"  

def get_num_sequences_in_a2m(filepath):
    """Counts the number of sequences in an A2M file."""
    count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # In A2M/FASTA, sequence headers start with '>'
                if line.startswith('>'):
                    count += 1
    except FileNotFoundError:
        print(f"Error: MSA file not found at {filepath}")
        return 0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0
    return count

def determine_epochs(num_sequences):
    """Determines the number of epochs based on sequence count."""
    if num_sequences >= 800:
        return 3
    elif 500 <= num_sequences < 800:
        return 4
    elif 300 <= num_sequences < 500:
        return 5
    elif 100 <= num_sequences < 300:
        return 10
    else:
        # For less than 100 sequences
        return 20

def main():
    if not os.path.exists(FINETUNE_SCRIPT):
        print(f"Error: {FINETUNE_SCRIPT} not found in the current directory.")
        print("Please ensure finetune.py is in the same directory as main.py.")
        return

    dms_substitution_path = os.path.join(DMS_DIR, "DMS_substitutions.csv")
    if not os.path.exists(dms_substitution_path):
        print(f"Error: DMS_substitutions.csv not found at {dms_substitution_path}.")
        print(f"Please ensure {DMS_DIR} exists and contains DMS_substitutions.csv.")
        return

    try:
        dms_substitutions_df = pd.read_csv(dms_substitution_path)
    except Exception as e:
        print(f"Error reading DMS_substitutions.csv: {e}")
        return

    # Get all CSV files in the DMS directory, excluding DMS_substitutions.csv
    csv_files = glob.glob(os.path.join(DMS_DIR, "*.csv"))
    target_csv_files = [f for f in csv_files if os.path.basename(f) != "DMS_substitutions.csv"]

    if not target_csv_files:
        print(f"No target CSV files found in {DMS_DIR} (excluding DMS_substitutions.csv). Exiting.")
        return

    for csv_file_path in target_csv_files:
        csv_filename = os.path.basename(csv_file_path)
        print(f"\n--- Processing {csv_filename} ---")

        # Find the corresponding entry in DMS_substitutions.csv
        matching_row = dms_substitutions_df[dms_substitutions_df['DMS_filename'] == csv_filename]

        if matching_row.empty:
            print(f"No matching entry found for '{csv_filename}' in DMS_substitutions.csv. Skipping.")
            continue

        # Extract MSA_filename and DMS_id
        # .iloc[0] is used because filtering by 'DMS_filename' should ideally yield one row
        msa_filename = matching_row['MSA_filename'].iloc[0]
        dms_id = matching_row['DMS_id'].iloc[0]

        msa_filepath = os.path.join(MSA_FILES_DIR, msa_filename)
        if not os.path.exists(msa_filepath):
            print(f"MSA file not found: {msa_filepath}. Skipping finetuning for {csv_filename}.")
            continue

        # Get the number of sequences to determine epochs
        num_sequences = get_num_sequences_in_a2m(msa_filepath)
        if num_sequences == 0:
            print(f"No sequences found or error reading {msa_filepath}. Skipping finetuning for {csv_filename}.")
            continue

        epochs = determine_epochs(num_sequences)
        print(f"  Found {num_sequences} sequences in {msa_filename}. Setting epochs to {epochs}.")

        # Define and create the output directory
        output_dir = os.path.join("../models/", str(dms_id), "esm_ft")
        os.makedirs(output_dir, exist_ok=True)
        adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_model_path):
            print(f"  {adapter_model_path} already exists. Skipping finetuning for {csv_filename}.")
            continue
        print(f"  Output directory set to: {output_dir}")

        # Construct the command to run finetune.py
        command = [
            "python", FINETUNE_SCRIPT,
            "--msa-file", msa_filepath,
            "--output-dir", output_dir,
            "--esm-dir", ESM_MODEL_DIR,
            "--epochs", str(epochs)
            # You can add other arguments like --batch-size, --lr, --max-seqs if needed
            # For example, to set max-seqs to 1000 as per the finetune.py default handling:
            # "--max-seqs", "1000"
        ]

        print(f"  Executing command: {' '.join(command)}")
        try:
            # Use subprocess.run to execute the finetune.py script
            # check=True will raise an exception if the command returns a non-zero exit code
            subprocess.run(command, check=True)
            print(f"--- Successfully finished finetuning for {csv_filename}. ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running finetune.py for {csv_filename}: {e} ---")
            print(f"  Command failed with exit code {e.returncode}")
        except FileNotFoundError:
            print(f"--- Error: 'python' command not found. Ensure Python is in your system's PATH. ---")
        except Exception as e:
            print(f"--- An unexpected error occurred while running finetune.py for {csv_filename}: {e} ---")

if __name__ == "__main__":
    main()
    