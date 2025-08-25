#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
DMS_DIR="$SCRIPT_DIR/../DMS"
SCRIPT="$SCRIPT_DIR/run_probes.py"

SPLITS=("learning_curve" "position" "mutation" "regime" "score")

for csv in "$DMS_DIR"/*.csv; do
    DMS_NAME=$(basename "$csv" .csv)
    # Skip non-experiment CSVs
    if [[ "$DMS_NAME" == "DMS_substitutions" || "$DMS_NAME" == "README" ]]; then
        continue
    fi
    MODEL_PATH="$MODELS_DIR/$DMS_NAME"
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "[WARNING] Model directory not found for $DMS_NAME, skipping."
        continue
    fi
    
    for SPLIT in "${SPLITS[@]}"; do
        echo "[INFO] Running $DMS_NAME (split: $SPLIT)"
        python "$SCRIPT" --dataset "$DMS_NAME" --split_type "$SPLIT"
        if [[ $? -ne 0 ]]; then
            echo "[ERROR] Experiment failed for $DMS_NAME (split: $SPLIT)."
        fi
    done
done

