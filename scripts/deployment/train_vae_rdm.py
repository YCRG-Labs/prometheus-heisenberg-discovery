#!/usr/bin/env python3
"""VAE Training on RDM Features for Large System Sizes

This script trains a VAE on reduced density matrix (RDM) features
extracted from DMRG ground states. This enables scaling to L=6, 8, 10+
where full wavefunctions are inaccessible.

The RDM approach captures local quantum correlations while remaining
computationally tractable.

Usage:
    python train_vae_rdm.py --input groundstates_L6_rdm.h5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config


def setup_logger(output_dir: Path) -> logging.Logger:
    log_file = output_dir / f"vae_rdm_training_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class RDM_VAE(nn.Module):
    """VAE for RDM feature inputs.
    
    Architecture adapted for RDM features which are much smaller than
    full wavefunctions but still capture quantum correlations.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent mean (deterministic for inference)."""
        h = self.encoder(x)
        return self.fc_mu(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def loss_function(self, x: torch.Tensor, recon: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """RDM reconstruction loss + KL divergence.
        
        For RDMs, we use MSE reconstruction since they're real-valued
        density matrix elements (not wavefunctions requiring fidelity).
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - recon) ** 2)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


def load_rdm_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load RDM features from HDF5 file.
    
    Returns:
        Tuple of (rdm_features, j2_values, feature_dim)
    """
    features_list = []
    j2_values = []
    
    with h5py.File(filepath, 'r') as f:
        L = f.attrs.get('L', 0)
        
        for key in sorted(f.keys()):
            if not key.startswith('J2_'):
                continue
            
            grp = f[key]
            rdm_features = grp['rdm_features'][:]
            
            # Robustly determine j2_j1:
            # 1) Prefer group attribute "j2_j1" if present
            # 2) Fall back to dataset "j2_j1" inside the group
            # 3) Finally, parse it from the group name "J2_<value>"
            if 'j2_j1' in grp.attrs:
                j2_j1 = grp.attrs['j2_j1']
            elif 'j2_j1' in grp:
                j2_j1 = grp['j2_j1'][()]
            else:
                try:
                    j2_j1 = float(key.split('_', 1)[1])
                except Exception as exc:  # noqa: BLE001
                    raise KeyError(
                        f"Could not determine j2_j1 for group '{key}' in {filepath}"
                    ) from exc
            
            features_list.append(rdm_features)
            j2_values.append(j2_j1)
    
    features = np.array(features_list, dtype=np.float32)
    j2_values = np.array(j2_values)
    
    return features, j2_values, features.shape[1]


def train_rdm_vae(
    model: RDM_VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, List[float]]:
    """Train RDM VAE."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.max_epochs
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'val_recon': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(config.training.max_epochs):
        # Training
        model.train()
        train_losses = []
        train_recons = []
        
        for batch in train_loader:
            x = batch[0].to(device)
            
            recon, mu, logvar = model(x)
            loss_dict = model.loss_function(x, recon, mu, logvar, beta=config.training.beta)
            
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            
            train_losses.append(loss_dict['loss'].item())
            train_recons.append(loss_dict['recon_loss'].item())
        
        # Validation
        model.eval()
        val_losses = []
        val_recons = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon, mu, logvar = model(x)
                loss_dict = model.loss_function(x, recon, mu, logvar, beta=config.training.beta)
                val_losses.append(loss_dict['loss'].item())
                val_recons.append(loss_dict['recon_loss'].item())
        
        # Record history
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(np.mean(train_recons))
        history['val_recon'].append(np.mean(val_recons))
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{config.training.max_epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


def encode_all_states(
    model: RDM_VAE,
    features: np.ndarray,
    j2_values: np.ndarray,
    device: torch.device
) -> Dict[float, np.ndarray]:
    """Encode all states to latent space."""
    model.eval()
    latent_reps = {}
    
    with torch.no_grad():
        for i, j2 in enumerate(j2_values):
            # Avoid torch.from_numpy to work around environments where
            # Torch's NumPy bridge is disabled; go via Python lists instead.
            x_np = features[i : i + 1]
            x = torch.tensor(x_np.tolist(), dtype=torch.float32, device=device)
            z = model.encode(x)
            # Also avoid tensor.numpy() which relies on Torch's NumPy bridge.
            latent_reps[float(j2)] = np.asarray(z.detach().cpu().tolist(), dtype=np.float32).squeeze()
    
    return latent_reps


def parse_args():
    parser = argparse.ArgumentParser(description="Train RDM-based VAE")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file")
    parser.add_argument("--config", type=str, default=str(project_root / "configs" / "laptop_config.yaml"))
    parser.add_argument("--output_dir", type=str, default="results/vae_rdm")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir)
    logger.info("=" * 60)
    logger.info("RDM-based VAE Training")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    features, j2_values, feature_dim = load_rdm_data(args.input)
    logger.info(f"Loaded {len(j2_values)} states, feature dim = {feature_dim}")
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = RDM_VAE(
        input_dim=feature_dim,
        latent_dim=args.latent_dim,
        hidden_dims=[256, 128, 64]
    ).to(device)
    logger.info(f"Created RDM_VAE: input_dim={feature_dim}, latent_dim={args.latent_dim}")
    
    # Prepare data
    # Avoid torch.from_numpy to be robust when Torch's NumPy bridge is disabled.
    features_tensor = torch.tensor(features.tolist(), dtype=torch.float32)
    dataset = TensorDataset(features_tensor)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # Train
    logger.info("Starting training...")
    history = train_rdm_vae(model, train_loader, val_loader, config, device, logger)
    
    # Save model
    checkpoint_path = output_dir / "rdm_vae.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'feature_dim': feature_dim,
        'latent_dim': args.latent_dim,
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)
    logger.info(f"Saved model to {checkpoint_path}")
    
    # Encode all states
    logger.info("Encoding all states to latent space...")
    latent_reps = encode_all_states(model, features, j2_values, device)
    
    # Save latent representations
    latent_file = output_dir / "latent_representations.h5"
    with h5py.File(latent_file, 'w') as f:
        f.attrs['latent_dim'] = args.latent_dim
        f.attrs['n_states'] = len(latent_reps)
        for j2, z in latent_reps.items():
            key = f"J2_{j2:.3f}"
            f.create_dataset(key, data=z)
            f[key].attrs['j2_j1'] = j2
    logger.info(f"Saved latent representations to {latent_file}")
    
    # Report
    logger.info("Training complete!")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"  Final val loss: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
