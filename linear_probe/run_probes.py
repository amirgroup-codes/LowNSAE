#!/usr/bin/env python
import argparse, contextlib, os, re, sys, pickle, math, random, warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel

warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ──────────────────────────────────────────────────────────────────────────────
# Typing suffixes:
# B: Batch Size 
# L: Layer 
# T: Sequence length of protein (variable)
# D: SAE Latent Dimension 
# H: Hidden Dimension of ESM 
# N: Number of sequences (variable)
# V: ESM vocabulary size
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Constants
PLM_DIM   = 1280
SAE_DIM   = 4096
LAYER     = 24
BATCH_SIZE= 64
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.getcwd()))
from interprot.interprot.utils import get_layer_activations
from gen_utils import load_sae_model, get_logits_embeddings

def get_embeddings(seqs, esm_model, sae_model, tokenizer, device):
    esm_acts_N, sae_acts_N = [], []
    if device.type == "cuda": torch.cuda.empty_cache()
    for start in tqdm(range(0,len(seqs),BATCH_SIZE), desc="Computing embeddings", leave=False):
        batch_seqs_B = seqs[start:start+BATCH_SIZE]
        with torch.inference_mode():
            batch_acts_BTH = get_layer_activations(tokenizer, esm_model, batch_seqs_B, layer=LAYER, device=device)
        for acts_TH in batch_acts_BTH:
            core_TH = acts_TH[1:-1].to(device)
            if core_TH.numel()==0:
                esm_acts_N.append(torch.zeros(PLM_DIM,device=device).cpu().numpy())
                if sae_model: sae_acts_N.append(torch.zeros(SAE_DIM,device=device).cpu().numpy()); continue
            esm_acts_N.append(core_TH.mean(0).cpu().numpy())
            if sae_model:
                sae_vector_TD = sae_model.get_acts(core_TH)
                sae_acts_N.append(sae_vector_TD.mean(0).cpu().numpy())
    return np.stack(esm_acts_N), \
           (np.stack(sae_acts_N) if sae_model else None)

def get_logits_embeddings(seqs, esm_model, tokenizer, device):
    out_NV = [] # list length N, each entry is V dimensional
    if device.type=="cuda": torch.cuda.empty_cache()
    for start in tqdm(range(0,len(seqs),BATCH_SIZE),desc="Computing logits",leave=False):
        batch_B = seqs[start:start+BATCH_SIZE]
        inputs_BT = tokenizer(batch_B, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            logits_BTV = esm_model(**inputs_BT).logits
        for i in range(logits_BTV.size(0)): # loop length B
            seq_logits_TV = logits_BTV[i,1:-1] # drop CLS/EOS
            out_NV.append(seq_logits_TV.mean(0).cpu().numpy() if seq_logits_TV.numel() else
                       torch.zeros(esm_model.config.vocab_size).numpy())
    return np.vstack(out_NV)

def train_probe(X, y, alpha=1.0):
    scaler = StandardScaler().fit(X)
    model  = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    return model, scaler

def grid_search_ridge(X_train, y_train, X_val, y_val, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]):
    """Grid search for optimal ridge alpha using validation set."""
    best_alpha = alphas[0]
    best_score = -np.inf
    
    for alpha in alphas:
        model, scaler = train_probe(X_train, y_train, alpha=alpha)
        y_pred = model.predict(scaler.transform(X_val))
        score = r2_score(y_val, y_pred)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    model, scaler = train_probe(X_train, y_train, alpha=best_alpha)
    return model, scaler, best_alpha

def evaluate_probe(model, scaler, X, y):
    if len(y)<2: return math.nan, math.nan
    ypred = model.predict(scaler.transform(X))
    return r2_score(y, ypred), spearmanr(y, ypred)[0]

def save_weights_func(weights, seed, rerun, train_size, model_name, ridge_model=None, scaler=None, output_dir=None):
    if output_dir is None or weights is None: return
    run_dir = output_dir / f"probe_weights/seed{seed}_rerun{rerun}_train{train_size}"
    run_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(sorted(enumerate(weights), key=lambda x:abs(x[1]), reverse=True),
                 columns=["Index","Weight"]).to_csv(run_dir/f"model_{model_name}.csv", index=False)
    if ridge_model and scaler:
        with open(run_dir/f"model_{model_name}.pkl",'wb') as f:
            pickle.dump({'ridge_model':ridge_model,'scaler':scaler},f)
    print(f"Saved weights to {run_dir/f'model_{model_name}.csv'}")

AA_RE = re.compile(r"([A-Z])(\d+)([A-Z])") 

def _parse_subs(s):
    if not isinstance(s,str) or not s: return []
    muts=[]
    for tok in re.split(r"[;, ]+", s.strip()):
        m = AA_RE.fullmatch(tok)
        if m: muts.append((int(m.group(2)), m.group(1), m.group(3)))  # (pos, wt, mut)
    return muts

def make_split_indices(df, split_type, rng, test_frac=0.2, train_sizes=None):
    n=len(df)
    idx=np.arange(n)
    if split_type=="learning_curve":           # handled elsewhere
        raise RuntimeError("learning_curve split handled separately")
    elif split_type == "score":
        if 'DMS_score_bin' not in df.columns:
            raise RuntimeError("Column 'DMS_score_bin' required for score split")
        train_idx = df[df['DMS_score_bin'] == 0].index.values
        test_idx  = df[df['DMS_score_bin'] == 1].index.values
    else:
        # Need mutation metadata
        if 'Substitutions' in df.columns:
            muts = df['Substitutions'].apply(_parse_subs)
        else:
            raise RuntimeError(f"Column 'Substitutions' required for {split_type} split")
        df = df.assign(_muts=muts,
                       _positions=muts.apply(lambda l: {p for p,_,_ in l}),
                       _sub_tokens=muts.apply(lambda l: {f"{p}{a}{b}" for p,a,b in l}),
                       _num_muts = muts.apply(len))
        if split_type=="position":
            all_pos = sorted({p for s in df._positions for p in s})
            rng.shuffle(all_pos)
            cut=int(len(all_pos)*test_frac)
            test_pos=set(all_pos[:cut]); train_pos=set(all_pos[cut:])
            print(f"[DEBUG] Position split: {len(all_pos)} total positions, {len(test_pos)} test positions, {len(train_pos)} train positions")
            
            def in_set(pos_set, allowed): 
                return pos_set.issubset(allowed) and len(pos_set)>0
            train_idx=df[df._positions.apply(in_set,allowed=train_pos)].index.values
            test_idx =df[df._positions.apply(in_set,allowed=test_pos)].index.values
            
            if len(test_idx) < 10:
                print(f"80/20 split didn't work, this should only happen for SPG1_STRSG_Wu_2016")
                rng.shuffle(all_pos)
                split_point = int(len(all_pos) * 0.75)  
                train_pos = set(all_pos[:split_point])
                test_pos = set(all_pos[split_point:])
                train_idx = df[df._positions.apply(in_set, allowed=train_pos)].index.values
                test_idx = df[df._positions.apply(in_set, allowed=test_pos)].index.values
                print(f"[DEBUG] Alternative position split: {len(train_idx)} train samples, {len(test_idx)} test samples")
        elif split_type=="mutation":
            all_tok = sorted({tok for s in df._sub_tokens for tok in s})
            rng.shuffle(all_tok); cut=int(len(all_tok)*test_frac)
            test_tok=set(all_tok[:cut]); train_tok=set(all_tok[cut:])
            def tok_ok(tokset, allowed): return tokset.issubset(allowed) and len(tokset)>0
            train_idx=df[df._sub_tokens.apply(tok_ok,allowed=train_tok)].index.values
            test_idx =df[df._sub_tokens.apply(tok_ok,allowed=test_tok)].index.values
        elif split_type=="regime":
            max_k=df._num_muts.max()
            if max_k<=2:
                train_idx=df[df._num_muts==1].index.values      # singles
                test_idx =df[df._num_muts==2].index.values      # doubles
            else:
                train_idx=df[df._num_muts<=2].index.values     
                test_idx =df[df._num_muts>2].index.values       
                if len(train_idx) < max(train_sizes):
                    print("Training on singles, doubles, and triples, should only be for F7YBW8_MESOW_Ding_2023")
                    train_idx=df[df._num_muts<=3].index.values
                    test_idx =df[df._num_muts>3].index.values
                    print(f"{len(train_idx)} train samples, {len(test_idx)} test samples")
        else: 
            raise ValueError(split_type)
    rng.shuffle(train_idx)
    val_cut = max(1,int(0.1*len(train_idx)))
    val_idx = train_idx[:val_cut]
    train_idx=train_idx[val_cut:]
    
    #final check: ensure we have enough test samples
    if len(test_idx) < 2:
        print(f"[ERROR] {split_type} split: test set has only {len(test_idx)} samples, which is too small for evaluation")
        #return empty arrays to indicate failure
        return np.array([]), np.array([]), np.array([])
    
    return train_idx, val_idx, test_idx

def main():
    parser=argparse.ArgumentParser(description="Run ridge probes with multiple split types")
    parser.add_argument("--dataset", required=True,
                        help="CSV stem, e.g. GFP_AEQVI_Sarkisyan_2016")
    parser.add_argument("--train_sizes", type=int, nargs="+",
                        default=[6,8,24,96,384], help="Only for learning_curve")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_reruns", type=int, default=3)
    parser.add_argument("--holdout_size", type=float, default=0.1,
                        help="for learning_curve random test split")
    parser.add_argument("--split_type", choices=["learning_curve","position","mutation","regime","score"],
                        default="learning_curve")
    parser.add_argument("--root_dir", default="..",
                        help="Top‑level folder containing DMS/ and models/")

    args=parser.parse_args()

    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

    root     = Path(args.root_dir)
    data_csv = root/'DMS'/f"{args.dataset}.csv"
    model_dir= root/'models'/args.dataset
    dataset_dir   = root/'linear_probe'/'results'/args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_workdir = dataset_dir/args.split_type
    split_workdir.mkdir(parents=True, exist_ok=True)
    # shared cache across all split types
    emb_cache_dir = dataset_dir/'emb_cache'
    emb_cache_dir.mkdir(parents=True, exist_ok=True)
    (split_workdir/'probe_weights').mkdir(exist_ok=True)

    df = pd.read_csv(data_csv)

    if "Substitutions" not in df.columns:
        df["Substitutions"] = df["mutant"].astype(str).str.replace(":", ";")

    muts = df['Substitutions'].apply(_parse_subs)
    positions = muts.apply(lambda l: {p for p,_,_ in l})
    print("unique mutated positions:", len({p for s in positions for p in s}))
    seqs   = df['mutated_sequence'].tolist()
    scores = df['DMS_score'      ].to_numpy()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=True
    esm_base_path = root/'models'/'esm2_t33_650M_UR50D'
    esm_ft_path   = model_dir/'esm_ft'
    sae_path = model_dir/'sae/l24_dim4096_k128_auxk256/checkpoints/last.ckpt'
    tok = AutoTokenizer.from_pretrained(str(esm_base_path))
    #base_esm = AutoModelForMaskedLM.from_pretrained(str(esm_base_path)).to(device).eval()
    ft_esm   = PeftModel.from_pretrained(
                   AutoModelForMaskedLM.from_pretrained(str(esm_base_path)),
                   str(esm_ft_path)).to(device).eval()
    sae_model = load_sae_model(str(sae_path), PLM_DIM, SAE_DIM, device)
    MODEL_REGISTRY = {
        "ft"      : (ft_esm, None),
        "sae"  : (ft_esm, sae_model),
        "ft_logits": (ft_esm, "logits")
    }
    @lru_cache(maxsize=None)
    def cached_embeddings(key):
        cache = emb_cache_dir/f"{key}.npz"
        mdl,sae=MODEL_REGISTRY[key]
        if sae=="logits":
            expect_dim=mdl.config.vocab_size
            if cache.exists():
                feats=np.load(cache)['feats']; 
                if feats.shape[1]==expect_dim:
                    return feats
            feats=get_logits_embeddings(seqs, mdl, tok, device)
            np.savez_compressed(cache, feats=feats)
            return feats
        else:
            expect_dim=SAE_DIM if sae else PLM_DIM
            if cache.exists():
                feats=np.load(cache)['feats']
                if feats.shape[1]==expect_dim:
                    return feats
            esm_arr, sae_arr = get_embeddings(seqs, mdl, sae, tok, device)
            feats = sae_arr if sae is not None else esm_arr
            np.savez_compressed(cache, feats=feats)
            return feats
    for k in MODEL_REGISTRY:
        print("model", k)
        _ = cached_embeddings(k)
    results=[]
    split_type=args.split_type
    
    if split_type in ["position", "mutation", "regime"] and "Substitutions" not in df.columns:
        print(f"[WARNING] Split type '{split_type}' requires 'Substitutions' column but dataset doesn't have it. Skipping.")
        return
    
    for seed in tqdm(range(args.n_seeds), desc="Seeds", leave=True):
        rng = np.random.RandomState(seed)
        if split_type=="learning_curve":
            holdout_idx=rng.choice(len(seqs), int(len(seqs)*args.holdout_size), replace=False)
            train_pool = np.setdiff1d(np.arange(len(seqs)), holdout_idx)
            for rerun in range(args.n_reruns):
                rng_rer = np.random.RandomState(seed+rerun*1000)
                for train_size in args.train_sizes:
                    if train_size>len(train_pool): continue
                    tr_idx = rng_rer.choice(train_pool, train_size, replace=False)
                    te_idx = holdout_idx
                    val_idx = None
                    if len(tr_idx) >= 2:
                        val_frac = 0.2 if split_type == "regime" else 0.1
                        val_size = max(1, int(val_frac * len(tr_idx)))
                        rng_rer.shuffle(tr_idx)          
                        val_idx = tr_idx[:val_size]
                        tr_idx  = tr_idx[val_size:]
                    for key in tqdm(MODEL_REGISTRY, desc=f"Models (train_size={train_size})", leave=False):
                        feats=cached_embeddings(key)
                        X_tr, y_tr = feats[tr_idx], scores[tr_idx]
                        X_te, y_te = feats[te_idx], scores[te_idx]
                        if val_idx is not None and len(val_idx) > 0:
                            X_val, y_val = feats[val_idx], scores[val_idx]
                            model, scaler, alpha_used = grid_search_ridge(X_tr, y_tr, X_val, y_val, alphas)
                        else:
                            model, scaler = train_probe(X_tr, y_tr)
                            alpha_used = 1.0
                        tr_r2,tr_p=evaluate_probe(model,scaler,X_tr,y_tr)
                        te_r2,te_p=evaluate_probe(model,scaler,X_te,y_te)
                        results.append(dict(seed=seed,rerun=rerun,train_size=train_size,
                                            split=split_type,model=key,
                                            train_r2=tr_r2,train_p=tr_p,
                                            test_r2=te_r2 ,test_p= te_p,
                                            alpha=alpha_used))
                        if seed==0 and rerun==0:
                            save_weights_func(model.coef_,seed,rerun,train_size,key,
                                              ridge_model=model,scaler=scaler,output_dir=split_workdir)
        else:
            tr_idx,val_idx,te_idx = make_split_indices(df, split_type, rng, train_sizes=args.train_sizes)
            if len(tr_idx) == 0 or len(te_idx) == 0:
                print(f"[ERROR] {split_type} split failed - insufficient data. Skipping this split type.")
                continue
            total_train_pool = len(tr_idx) + len(val_idx)
            print(f"[INFO] {split_type} split: {len(tr_idx)} train + {len(val_idx)} val = {total_train_pool} total train pool")
            print(f"[INFO] {split_type} split: {len(te_idx)} test points")
            max_requested = max(args.train_sizes)
            if max_requested > total_train_pool:
                print(f"[WARNING] Max requested train size ({max_requested}) > available train pool ({total_train_pool})")
            elif max_requested < total_train_pool * 0.1:  #less than 10% of available
                print(f"[INFO] Max requested train size ({max_requested}) is much smaller than available train pool ({total_train_pool})")
            train_pool = np.concatenate([tr_idx, val_idx])
            test_idx = te_idx
            print(f"[INFO] Data split sizes: train pool = {len(train_pool)}, test = {len(test_idx)}, val = {len(val_idx)}")
            for rerun in range(args.n_reruns):
                rng_rer = np.random.RandomState(seed+rerun*1000)
                for train_size in args.train_sizes:
                    if train_size > len(train_pool): 
                        print("ERROR: train_size > len(train_pool)")
                        continue
                    tr_idx = rng_rer.choice(train_pool, train_size, replace=False)
                    val_idx = None
                    if len(tr_idx) >= 2:
                        val_frac = 0.2 if split_type == "regime" else 0.1
                        val_size = max(1, int(val_frac * len(tr_idx)))
                        rng_rer.shuffle(tr_idx)          
                        val_idx = tr_idx[:val_size]
                        tr_idx  = tr_idx[val_size:]
                    
                    for key in tqdm(MODEL_REGISTRY, desc=f"Models (train_size={train_size})", leave=False):
                        feats=cached_embeddings(key)
                        X_tr, y_tr = feats[tr_idx], scores[tr_idx]
                        X_te, y_te = feats[test_idx], scores[test_idx]
                        
                        #always do grid search if we have validation data
                        if val_idx is not None and len(val_idx) > 0:
                            X_val, y_val = feats[val_idx], scores[val_idx]
                            model, scaler, alpha_used = grid_search_ridge(X_tr, y_tr, X_val, y_val, alphas)
                        else:
                            model, scaler = train_probe(X_tr, y_tr)
                            alpha_used = 1.0
                        
                        tr_r2,tr_p=evaluate_probe(model,scaler,X_tr,y_tr)
                        te_r2,te_p=evaluate_probe(model,scaler,X_te,y_te)
                        
                        results.append(dict(seed=seed,rerun=rerun,train_size=train_size,
                                            split=split_type,model=key,
                                            train_r2=tr_r2,train_p=tr_p,
                                            test_r2=te_r2 ,test_p= te_p,
                                            alpha=alpha_used))
                        if seed==0 and rerun==0:
                            save_weights_func(model.coef_,seed,rerun,train_size,key,
                                              ridge_model=model,scaler=scaler,output_dir=split_workdir)

    res_df=pd.DataFrame(results)
    csv_path = dataset_dir / f"probe_{args.split_type}_results.csv"
    res_df.to_csv(csv_path, index=False)
    print("Saved to", csv_path)

if __name__=="__main__":
    main()
