import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from scoring_utils import MLP, one_hot_encode_sequence

# ──────────────────────────────────────────────────────────────────────────────
# Typing suffixes:
# L: Total number of amino acids, in this case 20
# T: Sequence length of protein (variable)
# N: Number of sequences (variable)
# E: Length of one-hot encoded sequence (L * T)
# B: Batch Size 
# ──────────────────────────────────────────────────────────────────────────────

DMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DMS"))
BATCH_SIZE = 512
MAX_EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dms_filename: str, output_file: str):
    """
    Trains an MLP model over the full sequence length for a given DMS file
    and saves the set of mutated positions.
    """
    print(f"[train_mlp.py] Script started for {dms_filename}") # <-- ADDED
    torch.random.manual_seed(42)
    dms_path = os.path.join(DMS_DIR, dms_filename)
    dms_id = os.path.basename(output_file).replace('.pt', '')

    try:
        dms_df = pd.read_csv(dms_path)
        print(f"[train_mlp.py] Loaded {dms_path}, found {len(dms_df)} rows.") # <-- ADDED
        if dms_df.empty or "mutated_sequence" not in dms_df.columns or dms_df["mutated_sequence"].isna().all():
            print(f"Skipping {dms_filename}: file is empty or lacks sequence data.")
            return
    except FileNotFoundError:
        print(f"Error: DMS file not found at {dms_path}")
        return

    # Determine full sequence length from the first valid sequence
    seq_len = len(dms_df["mutated_sequence"].iloc[0])

    print('-' * 24)
    print(f"Processing {dms_id} over full sequence length: {seq_len}")
    print(f"Using device: {DEVICE}")
    print(f"Model will be saved to: {output_file}")
    print('-' * 24)
    
    # Extract all unique mutated positions from the 'mutant' column
    all_mutants_str = ';'.join(dms_df['mutant'].dropna().astype(str))
    # Use a regex to find all numbers (positions) in the mutant strings
    valid_positions = set(map(int, re.findall(r'\d+', all_mutants_str)))
    print(f"Found {len(valid_positions)} unique mutated positions in the DMS file.")

    # Save the sorted list of valid_positions to a .npy file
    positions_output_path = output_file.replace('.pt', '.npy')
    np.save(positions_output_path, np.array(sorted(list(valid_positions))))
    print(f"Saved valid mutated positions to: {positions_output_path}")

    # One-hot encode the full sequences
    X_NE = np.stack([one_hot_encode_sequence(seq) for seq in dms_df["mutated_sequence"]])
    y_N = dms_df["DMS_score"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X_NE, y_N, test_size=0.2, random_state=42)
    
    print(f"[train_mlp.py] Data split complete:")
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Validation set size: {len(X_val)}")
    
    if len(X_train) == 0:
        print("[train_mlp.py] ERROR: Training set is empty after split. Cannot train model.")
        return

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MLP(input_dim=X_NE.shape[1]).to(DEVICE) #input dim = L * T 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_BE, y_B in train_loader:
            x_BE, y_B = x_BE.to(DEVICE), y_B.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_BE)
            loss = loss_fn(pred, y_B)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_BE, y_B in val_loader:
                x_BE, y_B = x_BE.to(DEVICE), y_B.to(DEVICE)
                pred = model(x_BE)
                val_loss += loss_fn(pred, y_B).item() * x_BE.size(0)
        val_loss /= len(val_loader.dataset)
        
        if epoch == 1 or epoch % 10 == 0: # Print periodically
            print(f"[{dms_id}] Epoch {epoch:03d}: Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model, output_file)
        print(f"Saved model for {dms_id} to {output_file}")
    else:
        print("Training did not improve, no model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model on a single DMS dataset.")
    parser.add_argument("--dms_filename", type=str, required=True, help="Filename of the DMS .csv file (e.g., 'GFP_AEQVI_Sarkisyan_2016.csv').")
    parser.add_argument("--output_file", type=str, required=True, help="Full path for the output model file (e.g., 'path/to/your/model.pt').")
    args = parser.parse_args()
    train_model(args.dms_filename, args.output_file)