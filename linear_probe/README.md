# Linear Probe Framework

This directory contains code for linear probing experiments on ESM2 and sparse autoencoders to predict protein properties from DMS data.

## Usage

Run all experiments:
```bash
sh run_all_expts.sh
```

Run a single experiment:
```bash
python run_probes.py --dataset <DMS_NAME> --split_type <SPLIT_TYPE>
```

You can also change the train sizes, number of seeds/reruns, and holdout size for the learning_curve split.
Details on each split can be found in Section 3.2 of the paper. 

## Results

Results are saved in the `results/` directory, organized by protein dataset (e.g., SPG1_STRSG_Wu_2016, GRB2_HUMAN_Faure_2021, etc.).

## Plotting

You can recreate the plots from the paper using the `figures.ipynb` file once the experiments have been run.
