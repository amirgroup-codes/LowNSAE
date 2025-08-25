# Sparse Autoencoders Improve Low - $N$ Protein Function Prediction and Design

This is the official code repository for the paper "Sparse Autoencoders Improve Low - $N$ Protein Function Prediction and Design", by Darin Tsui, Kunal Talreja, and Amirali Aghazadeh.


### Key Components:

1. **Sparse Autoencoder Training** (`sae_training/`): Trains SAEs on ESM model activations
2. **Linear Probe Analysis** (`linear_probe/`): Evaluates SAE representations for fitness prediction
3. **Protein Engineering Pipeline** (`protein_engineering/`): Design sequences with SAE and ESM2
4. **Visualization Tools** (`visualize_sequences/`): Analysis and visualization of results

### Environment Setup

1. **Create conda environment:**
   ```bash
   conda env create -f sae.yml
   conda activate sae
   ```

### Running Experiments

1. **Train SAE models:**
   ```bash
   cd sae_training
   python main.py
   ```

2. **Run linear probe experiments:**
   ```bash
   cd linear_probe
   sh run_all_expts.sh
   ```

3. **Execute protein engineering pipeline:**
   ```bash
   cd protein_engineering
   python main.py
   ```

4. **Visualize results:**
   ```bash
   cd visualize_sequences
   python main.py
   ```

For the functions involving tensors, we use shape suffixes to describe their shape.

## Experiments

The project evaluates performance across multiple Deep Mutational Scanning (DMS) datasets from ProteinGym:

- **GFP_AEQVI_Sarkisyan_2016**
- **SPG1_STRSG_Olson_2014**
- **SPG1_STRSG_Wu_2016**
- **DLG4_HUMAN_Faure_2021**
- **GRB2_HUMAN_Faure_2021**
- **F7YBW8_MESOW_Ding_2023**

Each of these DMS files can be downloaded [here](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip). 

## Models Directory

This directory contains all the pre-trained models used in the Low-N Sparse Autoencoder (LowNSAE) framework for protein function prediction and design.

```
models/
├── esm2_t33_650M_UR50D/           # Base ESM2 protein language model
├── SPG1_STRSG_Wu_2016/            # Protein-specific models for SPG1
├── SPG1_STRSG_Olson_2014/         # Protein-specific models for SPG1 (Olson)
├── GRB2_HUMAN_Faure_2021/         # Protein-specific models for GRB2
├── GFP_AEQVI_Sarkisyan_2016/      # Protein-specific models for GFP
├── F7YBW8_MESOW_Ding_2023/        # Protein-specific models for F7YBW8
└── DLG4_HUMAN_Faure_2021/         # Protein-specific models for DLG4
```

Each protein directory contains two types of models:

#### Fine-tuned ESM2 Models (`esm_ft/`)
- `adapter_model.safetensors`: Fine-tuning adapter weights
- `adapter_config.json`: Adapter configuration
- `README.md`: Model card with detailed information
- `tokenizer_config.json`
- `special_tokens_map.json`
  
#### Sparse Autoencoders (`sae/`)
- `checkpoints/`: Training checkpoints with different loss values
- Model files with naming pattern: `esm2_plm{PLM_DIM}_l{SAE_LAYER}_sae{SAE_DIM}_k{SAE_K}_auxk{SAE_AUXK}_-step={step}-avg_mse_loss={loss}.ckpt`
- Last checkpoint simply named `last.ckpt`


## Usage

### Loading Fine-tuned Models

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load fine-tuned model
base_model = AutoModel.from_pretrained("models/esm2_t33_650M_UR50D")
ft_model = PeftModel.from_pretrained(base_model, f"models/{DMS_ASSAY}/esm_ft")
```

To run the experiments from the paper, simply create the `models` folder and train the ESM and SAE models. To run these experiments on your own protein, create the `DMS` folder and add a CSV of its DMS data, as well as the trained SAE/ESM model to `models`. The models used for this paper can be cloned from [HuggingFace](https://huggingface.co/ktalreja/LowNSAE).

