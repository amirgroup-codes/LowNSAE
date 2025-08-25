import os
import subprocess
import pandas as pd

DMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DMS"))
META_CSV = os.path.join(DMS_DIR, "DMS_substitutions.csv")
TRAIN_SCRIPT_PATH = "train_mlp.py"

def run_all_training():
    """
    Finds all DMS csv files, reads metadata, and runs the training script for each.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Load Metadata ---
    try:
        df_meta = pd.read_csv(META_CSV)
    except FileNotFoundError:
        print(f"FATAL ERROR: Metadata file not found at '{META_CSV}'.")
        return

    # --- Find DMS Files ---
    try:
        all_files = os.listdir(DMS_DIR)
        dms_files = sorted([f for f in all_files if f.endswith(".csv") and f != "DMS_substitutions.csv"])
    except FileNotFoundError:
        print(f"Error: DMS directory not found at '{DMS_DIR}'.")
        return

    if not dms_files:
        print("No DMS files found to process.")
        return

    print(f"Found {len(dms_files)} DMS files to process.")

    for dms_filename in dms_files:
        print(f"\n{'='*50}")
        print(f"[main.py] STARTING: {dms_filename}")
        print(f"{'='*50}")

        row = df_meta[df_meta["DMS_filename"] == dms_filename]
        if row.empty:
            print(f"[main.py] SKIPPING: No metadata found for {dms_filename} in {META_CSV}")
            continue
        
        dms_id = row["DMS_id"].values[0]
        output_file_path = os.path.join(script_dir, f"{dms_id}.pt")

        command = [
            "python", TRAIN_SCRIPT_PATH,
            "--dms_filename", dms_filename,
            "--output_file", output_file_path
        ]
        
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            print("--- Captured output from train_mlp.py ---")
            print(result.stdout)
            if result.stderr:
                print("--- Captured STDERR ---")
                print(result.stderr)
            print(f"--- [main.py] SUCCESSFULLY FINISHED: {dms_filename} ---")

        except FileNotFoundError:
            print(f"FATAL ERROR: The training script '{TRAIN_SCRIPT_PATH}' was not found.")
            break 
        except subprocess.CalledProcessError as e:
            print(f"--- [main.py] ERROR occurred while processing {dms_filename} ---")
            print(f"Return Code: {e.returncode}")
            print("\n--- STDOUT ---")
            print(e.stdout)
            print("\n--- STDERR ---")
            print(e.stderr)
            print(f"--- [main.py] FAILED: {dms_filename} ---")
            continue

    print(f"\n{'='*50}")
    print("All DMS processing complete.")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_all_training()