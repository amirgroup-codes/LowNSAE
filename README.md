# Sparse Autoencoders Improve Low - $N$ Protein Function Prediction and Design

This is the official code repository for the paper "Sparse Autoencoders Improve Low - $N$ Protein Function Prediction and Design", by Darin Tsui, Kunal Talreja, and Amirali Aghazadeh. A link to the paper can be found [here](https://arxiv.org/abs/2508.18567).


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

To run the experiments from the paper, simply create the `models` folder and train the ESM and SAE models. To run these experiments on your own protein, create the `DMS` folder and add a CSV of its DMS data, as well as the trained SAE/ESM model to `models`. The models used for this paper can be cloned from [HuggingFace](https://huggingface.co/ktalreja/LowNSAE), as well as a tutorial on how to use the models or setup your own.

