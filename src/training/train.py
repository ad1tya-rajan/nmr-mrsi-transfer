"""
Training harness for cross-scanner transfer model.

Data-agnostic training script that can work with synthetic or real data.
"""

import argparse
import yaml
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any

from ..models.mlp import create_model
from ..losses.huber import create_huber_loss
from ..utils.schema import default_schema, get_dim
from ..utils.normalization import fit_stats, transform, inverse_transform, TransformConfig
from ..utils.metrics import compute_all_metrics
from ..simulation.transfer_rules import create_paired_dataset


class ParameterDataset(Dataset):
    """Dataset for parameter pairs."""
    
    def __init__(
        self,
        theta_nmr: np.ndarray,
        theta_mrsi: np.ndarray,
        stats: Dict[str, np.ndarray] = None,
        transform_config: TransformConfig = None,
        normalize: bool = True
    ):
        """
        Args:
            theta_nmr: NMR parameters, shape (n_samples, D)
            theta_mrsi: MRSI parameters, shape (n_samples, D)
            stats: Normalization statistics (if None, will be computed)
            transform_config: Transform configuration
            normalize: Whether to normalize parameters
        """
        self.theta_nmr = theta_nmr
        self.theta_mrsi = theta_mrsi
        self.normalize = normalize
        
        if transform_config is None:
            transform_config = TransformConfig()
        self.transform_config = transform_config
        
        # Compute stats if not provided
        if stats is None and normalize:
            stats = fit_stats(theta_nmr, config=transform_config)
        self.stats = stats
        
        # Transform parameters
        if normalize:
            self.theta_nmr_t = transform(theta_nmr, self.stats, config=transform_config)
            self.theta_mrsi_t = transform(theta_mrsi, self.stats, config=transform_config)
        else:
            self.theta_nmr_t = theta_nmr
            self.theta_mrsi_t = theta_mrsi
    
    def __len__(self):
        return len(self.theta_nmr)
    
    def __getitem__(self, idx):
        return {
            "nmr": torch.FloatTensor(self.theta_nmr_t[idx]),
            "mrsi": torch.FloatTensor(self.theta_mrsi_t[idx])
        }


def create_synthetic_dataset(n_samples: int, config: Dict[str, Any]) -> ParameterDataset:
    """Create synthetic paired dataset."""
    transfer_kwargs = config.get("transfer", {})
    theta_nmr, theta_mrsi = create_paired_dataset(n_samples, transfer_kwargs=transfer_kwargs)
    
    # Compute normalization stats
    transform_config = TransformConfig()
    stats = fit_stats(theta_nmr, config=transform_config)
    
    return ParameterDataset(
        theta_nmr, theta_mrsi,
        stats=stats,
        transform_config=transform_config,
        normalize=True
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        nmr = batch["nmr"].to(device)
        mrsi = batch["mrsi"].to(device)
        
        optimizer.zero_grad()
        pred = model(nmr)
        loss = criterion(pred, mrsi)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {"loss": total_loss / n_batches}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: Dict[str, np.ndarray],
    transform_config: TransformConfig
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_pred = []
    all_target = []
    
    with torch.no_grad():
        for batch in dataloader:
            nmr = batch["nmr"].to(device)
            mrsi = batch["mrsi"].to(device)
            
            pred = model(nmr)
            loss = criterion(pred, mrsi)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_pred.append(pred.cpu().numpy())
            all_target.append(mrsi.cpu().numpy())
    
    # Concatenate and inverse transform
    pred_norm = np.concatenate(all_pred, axis=0)
    target_norm = np.concatenate(all_target, axis=0)
    
    pred_phys = inverse_transform(pred_norm, stats, config=transform_config)
    target_phys = inverse_transform(target_norm, stats, config=transform_config)
    
    # Compute metrics
    metrics = compute_all_metrics(pred_phys, target_phys)
    metrics["val_loss"] = total_loss / n_batches
    
    return metrics


def train(
    config_path: str,
    output_dir: str = "experiments/default"
):
    """
    Main training function.
    
    Args:
        config_path: Path to config YAML file
        output_dir: Output directory for checkpoints and logs
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get parameter dimension
    D = get_dim()
    
    # Create model
    model_config = config["model"]
    model = create_model(
        input_dim=D,
        output_dim=D,
        hidden_dim=model_config.get("hidden_dim", 256),
        n_layers=model_config.get("n_layers", 4),
        dropout=model_config.get("dropout", 0.1)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    dataset_config = config["dataset"]
    if dataset_config["type"] == "synthetic":
        train_dataset = create_synthetic_dataset(
            dataset_config["n_train"],
            config
        )
        val_dataset = create_synthetic_dataset(
            dataset_config["n_val"],
            config
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0)
    )
    
    # Loss
    loss_config = config["loss"]
    criterion = create_huber_loss(
        delta=loss_config.get("delta", 1.0),
        weights=None  # Can add per-parameter weights here
    )
    
    # Optimizer
    optim_config = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optim_config.get("lr", 1e-3),
        weight_decay=optim_config.get("weight_decay", 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = None
    if "scheduler" in config:
        sched_config = config["scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_config.get("factor", 0.5),
            patience=sched_config.get("patience", 10)
        )
    
    # Training loop
    n_epochs = config["training"]["n_epochs"]
    best_val_loss = float("inf")
    
    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            train_dataset.stats, train_dataset.transform_config
        )
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_metrics["val_loss"])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
        print(f"  MAE Overall: {val_metrics['mae_overall']:.6f}")
        print(f"  Plausibility: {val_metrics['plausibility_ratio']:.4f}")
        
        # Save checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["val_loss"],
                "stats": train_dataset.stats,
                "config": config
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  Saved best model (val_loss={best_val_loss:.6f})")
    
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="experiments/default")
    args = parser.parse_args()
    
    train(args.config, args.output)

