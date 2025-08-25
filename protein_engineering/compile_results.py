import pandas as pd
import numpy as np
import os
import glob

# --- 1. Data Collection and Processing ---
def count_mutations(mutant_str):
    if pd.isna(mutant_str) or mutant_str == 'No mutations':
        return 0
    return mutant_str.count(':') + 1

RESULTS_DIR = "results"
file_order = [
    "sae_steering",
    "esm_ft_sa",
    "esm_ft_logits_sa",
    "random",
]

method_map = {
    "sae_steering": "SAE",
    "esm_ft_sa": "ESM embedding",
    "esm_ft_logits_sa": "ESM logits",
    "random": "Random",
}

DMS_DIR = os.path.join(RESULTS_DIR, "..", "..", "DMS")
csv_files = glob.glob(os.path.join(DMS_DIR, "*.csv"))
DMS_NAMES = [
    "GFP_AEQVI_Sarkisyan_2016",
    "SPG1_STRSG_Olson_2014",
    "SPG1_STRSG_Wu_2016",
    "DLG4_HUMAN_Faure_2021",
    "GRB2_HUMAN_Faure_2021",
    "F7YBW8_MESOW_Ding_2023"
]

results = {method: {} for method in method_map.values()}

# Define the desired order of suffixes for the console output
suffix_order = ["_5.csv", "_10.csv", "_4.csv"]

print(f"Starting analysis of files in '{RESULTS_DIR}'")

# --- Original Console Output Logic ---
for folder in DMS_NAMES:
    folder_path = os.path.join(RESULTS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    print("=" * 20)
    print(f"{folder}")
    print("=" * 20)

    for suffix in suffix_order:
        matching_files = [f for f in os.listdir(folder_path) if f.endswith(suffix)]
        if not matching_files:
            continue
        
        mutations = suffix.replace("_", "").replace(".csv", "")
        print("\n" + "#" * 15)
        print(f"Mutations away: {mutations}")
        print("#" * 15 + "\n")

        ordered_csv_files = []
        for prefix in file_order:
            file_name = prefix + suffix
            if file_name in matching_files:
                ordered_csv_files.append(file_name)

        for csv_file in ordered_csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(csv_path)

            print(f"CSV file: {csv_file}")

            # Merge DMS_score onto df if it's the specific file
            if folder == "SPG1_STRSG_Wu_2016":
                dms_df_path = os.path.join(DMS_DIR, f"{folder}.csv")
                dms_df = pd.read_csv(dms_df_path)
                merged_df = pd.merge(df, dms_df[['mutant', 'DMS_score']], on='mutant', how='left')
                score_column = "DMS_score"
            else:
                merged_df = df
                score_column = "MLP_score"
            
            mean_score = merged_df[score_column].mean()
            std_score = merged_df[score_column].std()
            max_score = merged_df[score_column].max()
            
            sorted_df = merged_df.sort_values(by=score_column, ascending=False)
            print(sorted_df.head())
            
            top_10_percent_count = int(len(sorted_df) * 0.1)
            top_10_percent = sorted_df[score_column].head(top_10_percent_count)
            mean_10_percent = top_10_percent.mean() if top_10_percent_count > 0 else np.nan
            std_10_percent = top_10_percent.std() if top_10_percent_count > 0 else np.nan
            
            top_20_percent_count = int(len(sorted_df) * 0.2)
            top_20_percent = sorted_df[score_column].head(top_20_percent_count)
            mean_20_percent = top_20_percent.mean() if top_20_percent_count > 0 else np.nan
            std_20_percent = top_20_percent.std() if top_20_percent_count > 0 else np.nan

            print(f" - Average score: {mean_score:.2f} +/- {std_score:.2f}")
            print(f" - Top score: {max_score:.2f}")
            if not np.isnan(mean_10_percent):
                print(f" - Top 10% score: {mean_10_percent:.2f} +/- {std_10_percent:.2f}")
            if not np.isnan(mean_20_percent):
                print(f" - Top 20% score: {mean_20_percent:.2f} +/- {std_20_percent:.2f}")
            
            df['num_mutations'] = df['mutant'].apply(count_mutations)
            avg_num_mutations = df['num_mutations'].mean()
            print(f" - Average number of mutations: {avg_num_mutations:.2f}")

            # Store results for LaTeX table generation (only for mutation radius 5)
            if suffix == "_5.csv" or suffix == "_4.csv":
                prefix = csv_file.replace(suffix, '')
                method_name = method_map[prefix]
                if method_name not in results:
                    results[method_name] = {}
                results[method_name][folder] = {
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "max_score": max_score,
                    "mean_10_percent": mean_10_percent,
                    "std_10_percent": std_10_percent,
                    "mean_20_percent": mean_20_percent,
                    "std_20_percent": std_20_percent
                }

# --- 2. LaTeX Table Generation ---
latex_output = []
latex_output.append(r"\begin{table*}[t]")
latex_output.append(r"\vspace{0.2cm}")
latex_output.append(r"\caption{Mutation radius of 5}")
latex_output.append(r"\label{tab:central}")
latex_output.append(r"\centering")
latex_output.append(r"\resizebox{1 \textwidth}{!}{%")
latex_output.append(r"\begin{tabular}{lrrrrrr}")
latex_output.append(r"\toprule")
latex_output.append(r"\textbf{Method} & \textbf{DMS}")
latex_output.append(r"  & \textbf{Mean fitness $\uparrow$} & \textbf{Max fitness $\uparrow$}")
latex_output.append(r"  & \textbf{Top 10\% fitness $\uparrow$} & \textbf{Top 20\% fitness $\uparrow$} \\")
latex_output.append(r"\midrule")

first_method = True
for method_name in file_order:
    method_key = method_map[method_name]
    if method_key not in results:
        continue

    is_first_dms = True
    for dms in DMS_NAMES:
        if dms in results[method_key]:
            data = results[method_key][dms]
            
            max_dms_values = {
                "mean_score": -np.inf, "max_score": -np.inf,
                "mean_10_percent": -np.inf, "mean_20_percent": -np.inf
            }
            for other_method_key in method_map.values():
                if dms in results[other_method_key]:
                    other_data = results[other_method_key][dms]
                    max_dms_values["mean_score"] = max(max_dms_values["mean_score"], other_data["mean_score"])
                    max_dms_values["max_score"] = max(max_dms_values["max_score"], other_data["max_score"])
                    max_dms_values["mean_10_percent"] = max(max_dms_values["mean_10_percent"], other_data["mean_10_percent"])
                    max_dms_values["mean_20_percent"] = max(max_dms_values["mean_20_percent"], other_data["mean_20_percent"])

            dms_name_without_year = dms[:dms.rfind('_')]
            dms_name = dms_name_without_year.replace("_", r"\_")
            
            mean_score_str = f"{data['mean_score']:.2f}"
            if abs(data['mean_score'] - max_dms_values["mean_score"]) < 1e-6:
                mean_score_str = r"\textbf{" + mean_score_str + r"}"
            
            max_score_str = f"{data['max_score']:.2f}"
            if abs(data['max_score'] - max_dms_values["max_score"]) < 1e-6:
                max_score_str = r"\textbf{" + max_score_str + r"}"
            
            mean_10_percent_str = f"{data['mean_10_percent']:.2f}"
            if abs(data['mean_10_percent'] - max_dms_values["mean_10_percent"]) < 1e-6:
                mean_10_percent_str = r"\textbf{" + mean_10_percent_str + r"}"

            mean_20_percent_str = f"{data['mean_20_percent']:.2f}"
            if abs(data['mean_20_percent'] - max_dms_values["mean_20_percent"]) < 1e-6:
                mean_20_percent_str = r"\textbf{" + mean_20_percent_str + r"}"

            row_string = f"  & {dms_name}  & {mean_score_str} $\\pm$ {data['std_score']:.2f} & {max_score_str} & {mean_10_percent_str} $\\pm$ {data['std_10_percent']:.2f} & {mean_20_percent_str} $\\pm$ {data['std_20_percent']:.2f} \\\\"
            
            if is_first_dms:
                latex_output.append(r"\multirow{1}{*}{" + method_key + r"}")
                is_first_dms = False
            
            latex_output.append(row_string)
            
    latex_output.append(r"\cmidrule(lr){1-6}")

latex_output.append(r"\bottomrule")
latex_output.append(r"\end{tabular}%")
latex_output.append(r"}")
latex_output.append(r"\end{table*}")

# Write the output to a markdown file
output_file_path = "results.md"
with open(output_file_path, "w") as f:
    f.write("```latex\n")
    f.write("\n".join(latex_output))
    f.write("\n```")