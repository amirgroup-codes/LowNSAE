# Sequence Visualization Tools

This directory contains tools for analyzing and visualizing protein sequence data, particularly focusing on comparing wildtype and mutant sequences using Sparse Autoencoder (SAE) activations and linear probe weights.

## Usage

### Input Requirements
- **DMS data**: `../DMS/DMS_substitutions.csv` with protein metadata
- **Trained models**: SAE and fine-tuned ESM models in `../models/`
- **Probe weights**: Linear probe results in `../linear_probe/results/`
- **Mutant sequences**: Generated sequences in `../protein_engineering/results/`
- **Structure files**: PDB/CIF files in `pdbs/{protein_id}/`

### Running the Analysis
```bash
cd visualize_sequences
python main.py
```

Structure files are generated using the [AlphaFold server](https://alphafoldserver.com). The data from `pdbs`
was obtained from running AlphaFold on 8/10/2025.

### Output Structure
For each protein, the analysis generates:
```
results/{protein_id}/
├── highlighted_latents_layer{layer}.txt    # Wildtype analysis
└── {mutant_id}/
    ├── {mutant_id}.cif                     # Structure file
    ├── {mutant_id}_layer{layer}_analysis.txt  # Detailed comparison
    └── visualization files                 # Additional analysis outputs
```


### Output Format
Analysis files contain:
- **Wildtype section**: Top positive/negative latents and their peak locations
- **Mutant section**: Same analysis for mutant sequences
- **Difference section**: Significant changes between wildtype and mutant

