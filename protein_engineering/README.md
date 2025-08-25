# Protein Engineering Framework

This directory contains code for protein engineering experiments using various sequence generation methods including SAE latent steering, ESM embeddings, ESM logits, and random mutations.


## Usage

Run all protein engineering experiments:
```bash
python main.py
```

Run individual methods:
```bash
python steer_latents.py --dms_id <DMS_ID> --sae_path <SAE_PATH> --probe_model_path <PROBE_PATH> --mlp_model_path <MLP_PATH> --output_file <OUTPUT>
```

## Results

Results are saved in the `results/` directory, organized by protein dataset. 
