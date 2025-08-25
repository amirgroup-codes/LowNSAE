"""
Fine-tune ESM2 model on MSA using LoRA
"""
import os
import argparse
import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForMaskedLM

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
                
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProteinMSADataset(Dataset):
    def __init__(self, msa_file, max_seqs=None, max_length=1024, sample_method='random'):
        self.msa_file = msa_file
        self.max_length = max_length
        self.sample_method = sample_method
        self.seqs = self._read_a3m(msa_file)

        if max_seqs and len(self.seqs) > max_seqs:
            self.seqs = self._sample(self.seqs, max_seqs)

    def _read_a3m(self, msa_file):
        sequences = []
        with open(msa_file) as f:
            current = {"id": None, "seq": ""}
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current["id"]:
                        if len(current["seq"]) <= self.max_length:
                            sequences.append(current)
                    current = {"id": line[1:], "seq": ""}
                else:
                    current["seq"] += ''.join([c for c in line if c.isupper() or c == '-'])
            if current["id"] and len(current["seq"]) <= self.max_length:
                sequences.append(current)
        return sequences

    def _sample(self, seqs, max_seqs):
        first = seqs[0]
        others = random.sample(seqs[1:], max_seqs - 1)
        return [first] + others

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]["seq"]

def get_lora_targets(start_layer=0, end_layer=32):
    target_modules = []
    for i in range(start_layer, end_layer + 1):
        prefix = f"encoder.layer.{i}.attention.self"
        target_modules.extend([
            f"{prefix}.query",
            f"{prefix}.key",
            f"{prefix}.value"
        ])
    return target_modules

def mask_tokens(inputs, tokenizer, mask_prob=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    inputs[masked_indices] = tokenizer.mask_token_id
    return inputs, labels

def collate_batch(batch, tokenizer, mask_prob=0.15):
    encodings = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    input_ids = encodings["input_ids"]
    masked_inputs, labels = mask_tokens(input_ids, tokenizer, mask_prob)
    return masked_inputs, labels

def fine_tune(model, tokenizer, dataset, output_dir, batch_size=2, epochs=3, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda x: collate_batch(x, tokenizer))

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = [b.to(device) for b in batch]
            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")

    model.save_pretrained(os.path.join(output_dir))
    tokenizer.save_pretrained(os.path.join(output_dir))
    logger.info("Saved LoRA adapter and tokenizer.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--esm-dir", type=str, required=True)
    parser.add_argument("--max-seqs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-start", type=int, default=0, help="Start layer for LoRA injection")
    parser.add_argument("--lora-end", type=int, default=32, help="End layer for LoRA injection")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.esm_dir)
    model = AutoModelForMaskedLM.from_pretrained(args.esm_dir, trust_remote_code=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=get_lora_targets(args.lora_start, args.lora_end),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ProteinMSADataset(args.msa_file, max_seqs=args.max_seqs)
    os.makedirs(args.output_dir, exist_ok=True)

    fine_tune(model, tokenizer, dataset, args.output_dir,
              batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
