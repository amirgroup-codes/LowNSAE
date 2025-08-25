import argparse
import glob
import os

import pytorch_lightning as pl
import wandb
from data_module import SequenceDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sae_module import SAELightningModule

from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", type=str, default=None)
parser.add_argument("--esm2-weight", type=str, default="../models/esm2_t33_650M_UR50D.pt")
parser.add_argument("-l", "--layer-to_use", type=int, default=24)
parser.add_argument("--d-model", type=int, default=1280)
parser.add_argument("--d-hidden", type=int, default=4096)
parser.add_argument("-b", "--batch-size", type=int, default=48)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--k", type=int, default=128)
parser.add_argument("--auxk", type=int, default=256)
parser.add_argument("--dead-steps-threshold", type=int, default=2000)
parser.add_argument("-e", "--max-epochs", type=int, default=1)
parser.add_argument("-d", "--num-devices", type=int, default=1)
parser.add_argument("--model-suffix", type=str, default="")
parser.add_argument("--wandb-project", type=str, default="interprot")
parser.add_argument("--num-workers", type=int, default=None)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--cache-dir", type=str, default=".cache", 
                    help="Directory for cache files (wandb, home, etc.)")

args = parser.parse_args()

# Set up cache directories using user-provided cache directory
fake_home = os.path.join(args.cache_dir, "fake_home")
os.environ["HOME"] = fake_home
Path.home = lambda: Path(fake_home)

# Redirect all W&B data (cache, logs, artifacts, etc.)
os.environ["WANDB_DIR"] = os.path.join(args.cache_dir, ".wandb")
os.environ["WANDB_CACHE_DIR"] = os.path.join(args.cache_dir, ".wandb/cache")
os.environ["WANDB_CONFIG_DIR"] = os.path.join(args.cache_dir, ".wandb/config")
os.environ["WANDB_ARTIFACT_DIR"] = os.path.join(args.cache_dir, ".wandb/cache/artifacts")
os.environ["TMPDIR"] = os.path.join(args.cache_dir, ".wandb/tmp")

args.output_dir = os.path.join(args.output_dir, f"l{args.layer_to_use}_dim{args.d_hidden}_k{args.k}_auxk{args.auxk}")

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

sae_name = (
    f"esm2_plm1280_l{args.layer_to_use}_sae{args.d_hidden}_"
    f"k{args.k}_auxk{args.auxk}_{args.model_suffix}"
)
wandb_logger = WandbLogger(
    project=args.wandb_project,
    name=sae_name,
    save_dir=os.path.join(args.output_dir, "wandb"),
)

model = SAELightningModule(args)
wandb_logger.watch(model, log="all")

data_module = SequenceDataModule(args.data_dir, args.batch_size, args.num_workers)
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.output_dir, "checkpoints"),
    filename=sae_name + "-{step}-{avg_mse_loss:.2f}",
    save_top_k=3,
    monitor="train_loss",
    mode="min",
    save_last=True,
)


trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="gpu",
    devices=list(range(args.num_devices)),
    strategy="auto",
    logger=wandb_logger,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,  
    limit_val_batches=10,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

trainer.fit(model, data_module)
trainer.test(model, data_module)

for checkpoint in glob.glob(os.path.join(args.output_dir, "checkpoints", "*.ckpt")):
    ckpt_path = Path(checkpoint)
    artifact_name = ckpt_path.name.replace("=", "-").replace(" ", "_")
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)

wandb.finish()