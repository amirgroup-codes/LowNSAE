import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Typing suffixes:
# T: Sequence length of protein (variable)
# D: SAE Latent Dimension 
# ──────────────────────────────────────────────────────────────────────────────

def extract_top_peaks(weights_df, activations, top_k=5):
    """
    Extracts the peak activation locations for the top k positive and negative latents.
    """
    sorted_df = weights_df.sort_values("Weight", ascending=False)
    top_pos, top_neg = [], []

    # Find top k positive latents that are active
    for _, row in sorted_df.iterrows():
        if len(top_pos) == top_k: break
        idx = int(row["Index"])
        if row["Weight"] > 0 and np.any(activations[:, idx] != 0):
            top_pos.append(idx)

    # Find top k negative latents that are active
    for _, row in sorted_df.iloc[::-1].iterrows():
        if len(top_neg) == top_k: break
        idx = int(row["Index"])
        if row["Weight"] < 0 and np.any(activations[:, idx] != 0):
            top_neg.append(idx)

    unit_peaks = {}
    for idx in top_pos:
        peaks, _ = find_peaks(activations[:, idx], height=0)
        if len(peaks) > 0:
            unit_peaks[(idx, "Positive")] = sorted((peaks + 1).tolist())
    for idx in top_neg:
        peaks, _ = find_peaks(activations[:, idx], height=0)
        if len(peaks) > 0:
            unit_peaks[(idx, "Negative")] = sorted((peaks + 1).tolist())
    return unit_peaks


def analyze_activation_differences(sae_activations_wt, sae_activations_mut, sequence, weights_df, top_k=5, threshold=0.1, seq_offset=1):
    """
    Finds the top k positive and negative latents with the largest changes,
    and reports all per-residue changes for them that exceed a threshold.
    """
    diff_activations_TD = sae_activations_mut - sae_activations_wt
    resnum_to_aa = {r: sequence[r - seq_offset] for r in range(seq_offset, len(sequence) + seq_offset)}
    
    # Create a fast lookup for latent weights
    idx_to_weight = pd.Series(weights_df.Weight.values, index=weights_df.Index).to_dict()

    # Calculate the max absolute change for every latent
    max_abs_diffs_D = np.max(np.abs(diff_activations_TD), axis=0) # shape D_latent
    
    # Separate latents into positive and negative lists with their max change
    pos_latent_changes = []
    neg_latent_changes = []
    for idx, change in enumerate(max_abs_diffs_D):
        weight = idx_to_weight.get(idx)
        if weight is not None:
            if weight > 0:
                pos_latent_changes.append((change, idx))
            elif weight < 0:
                neg_latent_changes.append((change, idx))

    # Sort by the magnitude of the change (descending) and get top k indices
    pos_latent_changes.sort(key=lambda x: x[0], reverse=True)
    neg_latent_changes.sort(key=lambda x: x[0], reverse=True)
    
    top_pos_indices = [idx for change, idx in pos_latent_changes[:top_k]]
    top_neg_indices = [idx for change, idx in neg_latent_changes[:top_k]]

    diff_lines = {"Positive": [], "Negative": []}

    # Process and format the top 5 positive latents
    for idx in top_pos_indices:
        peaks_pos, props_pos = find_peaks(diff_activations_TD[:, idx], height=threshold) # list of length n_peaks
        peaks_neg, props_neg = find_peaks(-diff_activations_TD[:, idx], height=threshold) # list of length n_peaks
        
        if len(peaks_pos) > 0 or len(peaks_neg) > 0:
            change_tags = []
            for p, h in zip(peaks_pos, props_pos['peak_heights']):
                change_tags.append(f"{resnum_to_aa.get(p + seq_offset, 'X')}{p + seq_offset}(+{h:.2f})")
            for p, h in zip(peaks_neg, props_neg['peak_heights']):
                original_neg_val = diff_activations_TD[p, idx]
                change_tags.append(f"{resnum_to_aa.get(p + seq_offset, 'X')}{p + seq_offset}({original_neg_val:.2f})")
            
            line = f"Latent {idx}: " + ", ".join(change_tags)
            diff_lines["Positive"].append(line)

    # Process and format the top 5 negative latents
    for idx in top_neg_indices:
        peaks_pos, props_pos = find_peaks(diff_activations_TD[:, idx], height=threshold)
        peaks_neg, props_neg = find_peaks(-diff_activations_TD[:, idx], height=threshold)

        if len(peaks_pos) > 0 or len(peaks_neg) > 0:
            change_tags = []
            for p, h in zip(peaks_pos, props_pos['peak_heights']):
                change_tags.append(f"{resnum_to_aa.get(p + seq_offset, 'X')}{p + seq_offset}(+{h:.2f})")
            for p, h in zip(peaks_neg, props_neg['peak_heights']):
                original_neg_val = diff_activations_TD[p, idx]
                change_tags.append(f"{resnum_to_aa.get(p + seq_offset, 'X')}{p + seq_offset}({original_neg_val:.2f})")

            line = f"Latent {idx}: " + ", ".join(change_tags)
            diff_lines["Negative"].append(line)

    return diff_lines

def format_peaks_for_output(unit_peaks, wt_sequence, seq_offset=1):
    """ Formats the extracted peaks into a list of strings for writing to a file. """
    resnum_to_aa = {r: wt_sequence[r - seq_offset] for r in range(seq_offset, len(wt_sequence) + seq_offset)}
    output_lines = {"Positive": [], "Negative": []}
    sorted_peaks = sorted(unit_peaks.items(), key=lambda item: item[0][0])
    for (unit, sign), residues in sorted_peaks:
        aa_tags = [f"{resnum_to_aa.get(r, 'X')}{r}" for r in residues]
        output_lines[sign].append(f"Latent {unit}: " + ", ".join(aa_tags))
    return output_lines

def generate_visualization_files(af_id, top_mutant_id, diff_output_lines, mutant_output_dir):

    pml_script_filename = f"{af_id}.pml"
    bfactor_filenames = {
        "pos_increase": f"{af_id}_pos_increase.txt",
        "pos_decrease": f"{af_id}_pos_decrease.txt",
        "neg_increase": f"{af_id}_neg_increase.txt",
        "neg_decrease": f"{af_id}_neg_decrease.txt",
    }

    bfactor_data = {key: {} for key in bfactor_filenames}

    def parse_and_sort_changes(lines, latent_type):
        for line in lines:
            matches = re.findall(r"([A-Z])(\d+)\(([-+\d\.]+)\)", line)
            for _, resi, value_str in matches:
                resi, value = int(resi), float(value_str)
                if latent_type == "Positive" and value > 0:
                    category = "pos_increase"
                elif latent_type == "Positive" and value < 0:
                    category = "pos_decrease"
                elif latent_type == "Negative" and value > 0:
                    category = "neg_increase"
                else:
                    category = "neg_decrease"
                bfactor_data[category][resi] = max(
                    bfactor_data[category].get(resi, 0), abs(value)
                )

    parse_and_sort_changes(diff_output_lines["Positive"], "Positive")
    parse_and_sort_changes(diff_output_lines["Negative"], "Negative")

    for category, filename in bfactor_filenames.items():
        with open(mutant_output_dir / filename, "w") as f:
            for resi, value in sorted(bfactor_data[category].items()):
                f.write(f"{resi} {value}\n")

    pml_content = f"""
# PyMOL Script for 4-Way Analysis of Mutant: {top_mutant_id}
reinitialize
load {af_id}.cif, original
hide everything
show cartoon, original
set_color new_color, [0.106, 0.459, 0.733]
color new_color, original
remove solvent

set_color dark_red_custom, [0.6, 0.0, 0.0]

python
import os
from pymol import cmd

def color_residues(object_name, bfactor_file):
    if not os.path.exists(bfactor_file):
        cmd.do(f'print("B-factor file not found: {{bfactor_file}}")')
        return
    # Start new_color for everything
    cmd.color('new_color', object_name)
    resis = []
    with open(bfactor_file) as f:
        for line in f:
            if line.strip():
                resi, _ = line.split()
                resis.append(resi)
    if not resis:
        return
    selection = f"{{object_name}} and resi " + "+".join(resis)
    cmd.color('dark_red_custom', selection)
cmd.extend('color_residues', color_residues)
python end

create pos_increase_view, original
color_residues("pos_increase_view", "{bfactor_filenames['pos_increase']}")

create pos_decrease_view, original
color_residues("pos_decrease_view", "{bfactor_filenames['pos_decrease']}")

create neg_increase_view, original
color_residues("neg_increase_view", "{bfactor_filenames['neg_increase']}")

create neg_decrease_view, original
color_residues("neg_decrease_view", "{bfactor_filenames['neg_decrease']}")

delete original
bg_color white
set ray_trace_mode, 1
set ray_shadows, off
group all_views, pos_* neg_*
zoom all_views
"""

    with open(mutant_output_dir / pml_script_filename, "w") as f:
        f.write(pml_content)

    print(f"    -> Saved all analysis and visualization files to {mutant_output_dir}")