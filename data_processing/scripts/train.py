#!/usr/bin/env python
# /home/zdy/Project2/scripts/train.py

import yaml
import logging
import datetime
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

# --- Project-specific imports ---
from models.model import DynaModel
from models.loss import TotalLoss
from scripts.data import get_data_loader


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_cfg: dict, project_root: Path, log_base_name: str = "train"):
    """Configures logging for the run."""
    log_dir = project_root / log_cfg['log_dir']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = log_dir / f"{log_base_name}_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    log_filepath = run_log_dir / "run.log"
    log_level = getattr(logging, log_cfg.get('log_level', 'INFO').upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
        force=True
    )
    logging.info(f"Logging configured. Logs will be saved to: {run_log_dir}")
    return run_log_dir


def build_phy_geo_labels(batch: Batch, device: torch.device) -> dict:
    """Helper function to construct the phy_geo_labels dict for the loss function."""
    return {
        'bond': {'indices': batch.bond_indices.to(device), 'ref_lengths': batch.ref_bond_lengths.to(device)},
        'angle': {'indices': batch.angle_indices.to(device), 'ref_angles': batch.ref_angles.to(device)},
        'dihedral': {'indices': batch.dihedral_indices.to(device), 'true_angles': batch.true_dihedrals.to(device)},
        'vdw': {'indices': batch.vdw_indices.to(device), 'radii': batch.vdw_radii.to(device)},
        'electro': {'indices': batch.electro_indices.to(device)},
        'hbond': {'indices': batch.hbond_indices.to(device)},
        'pi_pi': {'ring_pair_indices': batch.pi_pi_ring_pair_indices},
        'partial_charges': batch.partial_charges.to(device)
    }


def evaluate(model: nn.Module, loader: PyGDataLoader, loss_fn: nn.Module, device: torch.device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss, total_rmsd = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(device)
            # For validation/testing, run the full diffusion refinement
            output = model(batch, use_diffusion_refinement=True)

            phy_geo_labels = build_phy_geo_labels(batch, device)

            loss_dict = loss_fn(
                pred_coords=output['pred_coords'], true_coords=output['true_coords'],
                pred_noise=output['pred_noise'], true_noise=output['true_noise'],
                phy_geo_labels=phy_geo_labels
            )

            total_loss += loss_dict['total_loss'].item()
            rmsd = torch.sqrt(torch.mean((output['pred_coords'] - output['true_coords']) ** 2))
            total_rmsd += rmsd.item()

    return total_loss / len(loader), total_rmsd / len(loader)


def main():
    """Main training and validation loop."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / 'config.yaml'
    config = load_config(config_path)

    project_root = config_path.parent
    train_cfg = config['training']

    # --- Initialization ---
    run_dir = setup_logging(config['logging']['training_log'], project_root)
    set_seed(train_cfg['seed'])
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Save a copy of the config for this run
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # --- Data Loading ---
    logging.info("Loading data...")
    train_loader = get_data_loader(config, 'train')
    val_loader = get_data_loader(config, 'val')
    if not train_loader or not val_loader:
        logging.critical("Failed to create data loaders. Exiting.")
        return

    logging.info(f"Train loader: {len(train_loader.dataset)} samples. Val loader: {len(val_loader.dataset)} samples.")

    # --- Model, Loss, Optimizer ---
    logging.info("Initializing model, loss function, and optimizer...")
    model = DynaModel(config).to(device)
    loss_fn = TotalLoss(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])

    # --- Training Loop ---
    best_val_loss = float('inf')
    logging.info("Starting training loop...")
    for epoch in range(train_cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")

        for i, batch in enumerate(progress_bar):
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch, use_diffusion_refinement=False)
            phy_geo_labels = build_phy_geo_labels(batch, device)

            loss_dict = loss_fn(
                pred_coords=output['pred_coords'], true_coords=output['true_coords'],
                pred_noise=output['pred_noise'], true_noise=output['true_noise'],
                phy_geo_labels=phy_geo_labels
            )

            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % train_cfg['log_every_n_steps'] == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        if (epoch + 1) % train_cfg['validate_every_n_epochs'] == 0:
            val_loss, val_rmsd = evaluate(model, val_loader, loss_fn, device)
            logging.info(f"Epoch {epoch + 1} | Validation Loss: {val_loss:.4f} | Validation RMSD: {val_rmsd:.4f}")

            # --- Save Best Model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dir = project_root / train_cfg['checkpoint_dir']
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = run_dir / f"best_model_epoch_{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, checkpoint_path)
                logging.info(f"New best model saved to {checkpoint_path}")

    logging.info("Training finished.")


if __name__ == '__main__':
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    main()