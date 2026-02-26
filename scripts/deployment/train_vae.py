#!/usr/bin/env python3
"""VAE Training Script - Reads from HDF5 Ground State Files

This script trains the Q-VAE on ground states stored in HDF5 format.
It's designed to run locally after downloading groundstates_L{L}.h5 from the VM.

The HDF5 file is the handoff point between:
- generate_groundstates.py (runs on VM, produces HDF5)
- train_vae.py (runs locally, consumes HDF5)

Usage:
    python train_vae.py --input groundstates_L6.h5 --config configs/laptop_config.yaml
    python train_vae.py --input results/groundstates/groundstates_L6.h5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.qvae_module import QVAE, QVAETrainer
from src.logging_config import setup_logging


def setup_logger(output_dir: Path) -> logging.Logger:
    """Setup logging for VAE training."""
    log_file = output_dir / f"vae_training_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_groundstates(filepath: str) -> Dict[float, Dict[str, Any]]:
    """Load ground states from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file (e.g., "groundstates_L6.h5")
        
    Returns:
        Dictionary mapping j2_j1 -> {
            "psi": wavefunction coefficients,
            "observables": array of 11 observable values,
            "energy": ground state energy,
            "L": lattice size,
            "chi": bond dimension used
        }
    """
    data = {}
    
    with h5py.File(filepath, "r") as f:
        for key in sorted(f.keys()):
            grp = f[key]
            
            j2_j1 = float(grp.attrs["j2_j1"])
            
            data[j2_j1] = {
                "psi": grp["psi"][:],
                "observables": grp["observables"][:],
                "energy": grp["energy"][()],
                "L": int(grp.attrs["L"]),
                "chi": int(grp.attrs.get("chi", 0))
            }
    
    return data


def get_hdf5_info(filepath: str) -> Dict[str, Any]:
    """Get information about HDF5 file contents.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Dictionary with file info
    """
    info = {
        "n_states": 0,
        "L": None,
        "j2_range": [],
        "hilbert_dim": None,
        "file_size_mb": Path(filepath).stat().st_size / (1024 * 1024)
    }
    
    with h5py.File(filepath, "r") as f:
        info["n_states"] = len(f.keys())
        
        for key in f.keys():
            grp = f[key]
            info["j2_range"].append(float(grp.attrs["j2_j1"]))
            
            if info["L"] is None:
                info["L"] = int(grp.attrs["L"])
                info["hilbert_dim"] = len(grp["psi"])
    
    info["j2_range"] = sorted(info["j2_range"])
    
    return info


def prepare_vae_dataset(
    groundstates: Dict[float, Dict[str, Any]],
    train_split: float = 0.8,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare train/validation dataloaders from ground states.
    
    Args:
        groundstates: Dictionary from load_groundstates()
        train_split: Fraction of data for training
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader, hilbert_dim)
    """
    # Convert wavefunctions to real vectors [Re(psi), Im(psi)]
    data_list = []
    j2_values = []
    
    for j2_j1, state_data in sorted(groundstates.items()):
        psi = state_data["psi"]
        
        # Convert complex to real: [Re(psi), Im(psi)]
        real_part = np.real(psi)
        imag_part = np.imag(psi)
        real_vector = np.concatenate([real_part, imag_part])
        
        data_list.append(real_vector)
        j2_values.append(j2_j1)
    
    # Stack into tensor
    data_array = np.array(data_list, dtype=np.float32)
    data_tensor = torch.from_numpy(data_array)
    
    # Get dimensions
    hilbert_dim = data_tensor.shape[1] // 2  # Half for real, half for imag
    
    # Create dataset
    dataset = TensorDataset(data_tensor)
    
    # Split into train/val
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, hilbert_dim


def train_qvae(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hilbert_dim: int,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[QVAE, Dict[str, Any]]:
    """Train Q-VAE model.
    
    Args:
        config: Configuration object
        train_loader: Training dataloader
        val_loader: Validation dataloader
        hilbert_dim: Hilbert space dimension
        device: Torch device
        logger: Logger instance
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Create model
    model = QVAE(config, hilbert_dim)
    model = model.to(device)
    
    logger.info(f"Created Q-VAE model:")
    logger.info(f"  Hilbert dim: {hilbert_dim}")
    logger.info(f"  Input dim: {2 * hilbert_dim}")
    logger.info(f"  Latent dim: {config.qvae_architecture.latent_dim}")
    logger.info(f"  Encoder layers: {config.qvae_architecture.encoder_layers}")
    logger.info(f"  Decoder layers: {config.qvae_architecture.decoder_layers}")
    logger.info(f"  Device: {device}")
    
    # Move data to device
    def move_batch_to_device(loader):
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                yield tuple(b.to(device) for b in batch)
            else:
                yield batch.to(device)
    
    # Create trainer
    trainer = QVAETrainer(model, config)
    
    # Train
    logger.info("Starting training...")
    logger.info(f"  Max epochs: {config.training.max_epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Beta (KL weight): {config.training.beta}")
    logger.info(f"  Patience: {config.training.patience}")
    
    history = trainer.train(train_loader, val_loader)
    
    return model, history


def save_model(
    model: QVAE,
    history: Dict[str, Any],
    output_dir: Path,
    L: int,
    config: Config
) -> Path:
    """Save trained model and history.
    
    Args:
        model: Trained Q-VAE model
        history: Training history
        output_dir: Output directory
        L: Lattice size
        config: Configuration used
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = output_dir / f"qvae_L{L}.pt"
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "L": L,
        "hilbert_dim": model.hilbert_dim,
        "latent_dim": model.latent_dim,
        "config": {
            "encoder_layers": config.qvae_architecture.encoder_layers,
            "decoder_layers": config.qvae_architecture.decoder_layers,
            "latent_dim": config.qvae_architecture.latent_dim,
            "beta": config.training.beta,
            "learning_rate": config.training.learning_rate,
        },
        "timestamp": datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def encode_all_states(
    model: QVAE,
    groundstates: Dict[float, Dict[str, Any]],
    device: torch.device
) -> Dict[float, np.ndarray]:
    """Encode all ground states to latent space.
    
    Args:
        model: Trained Q-VAE model
        groundstates: Dictionary from load_groundstates()
        device: Torch device
        
    Returns:
        Dictionary mapping j2_j1 -> latent vector
    """
    model.eval()
    latent_reps = {}
    
    with torch.no_grad():
        for j2_j1, state_data in sorted(groundstates.items()):
            psi = state_data["psi"]
            
            # Convert to real vector
            real_part = np.real(psi)
            imag_part = np.imag(psi)
            real_vector = np.concatenate([real_part, imag_part])
            
            # To tensor
            x = torch.from_numpy(real_vector.astype(np.float32)).unsqueeze(0).to(device)
            
            # Encode (deterministic - use mean)
            z = model.encode(x)
            
            latent_reps[j2_j1] = z.cpu().numpy().squeeze()
    
    return latent_reps


def save_latent_representations(
    latent_reps: Dict[float, np.ndarray],
    output_dir: Path,
    L: int
) -> Path:
    """Save latent representations to HDF5.
    
    Args:
        latent_reps: Dictionary mapping j2_j1 -> latent vector
        output_dir: Output directory
        L: Lattice size
        
    Returns:
        Path to saved file
    """
    output_file = output_dir / f"latent_representations_L{L}.h5"
    
    with h5py.File(output_file, "w") as f:
        f.attrs["L"] = L
        f.attrs["n_states"] = len(latent_reps)
        f.attrs["latent_dim"] = len(next(iter(latent_reps.values())))
        f.attrs["timestamp"] = datetime.now().isoformat()
        
        for j2_j1, z in latent_reps.items():
            key = f"J2_{j2_j1:.3f}"
            ds = f.create_dataset(key, data=z)
            ds.attrs["j2_j1"] = j2_j1
    
    return output_file


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Q-VAE on ground states from HDF5 file"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input HDF5 file with ground states (e.g., groundstates_L6.h5)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "laptop_config.yaml"),
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/vae_training",
        help="Output directory for model and results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max epochs (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda/mps, auto-detected if not specified)"
    )
    parser.add_argument(
        "--skip_encoding",
        action="store_true",
        help="Skip encoding states to latent space after training"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(output_dir)
    logger.info("=" * 60)
    logger.info("Q-VAE Training from HDF5 Ground States")
    logger.info("=" * 60)
    
    # Check input file
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Get file info
    info = get_hdf5_info(str(input_file))
    logger.info(f"Input file: {input_file}")
    logger.info(f"  States: {info['n_states']}")
    logger.info(f"  Lattice size: L={info['L']}")
    logger.info(f"  Hilbert dim: {info['hilbert_dim']}")
    logger.info(f"  J2 range: [{info['j2_range'][0]:.3f}, {info['j2_range'][-1]:.3f}]")
    logger.info(f"  File size: {info['file_size_mb']:.2f} MB")
    
    # Load config
    config = Config.from_yaml(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Override config if specified
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.max_epochs = args.epochs
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load ground states
    logger.info("Loading ground states from HDF5...")
    groundstates = load_groundstates(str(input_file))
    logger.info(f"Loaded {len(groundstates)} ground states")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    train_loader, val_loader, hilbert_dim = prepare_vae_dataset(
        groundstates,
        train_split=0.8,
        batch_size=config.training.batch_size
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train model
    model, history = train_qvae(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        hilbert_dim=hilbert_dim,
        device=device,
        logger=logger
    )
    
    # Save model
    L = info["L"]
    checkpoint_path = save_model(model, history, output_dir, L, config)
    logger.info(f"Saved model to {checkpoint_path}")
    
    # Report final metrics
    logger.info("Training complete:")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"  Final val loss: {history['val_loss'][-1]:.6f}")
    logger.info(f"  Final train fidelity: {history['train_fidelity'][-1]:.6f}")
    logger.info(f"  Final val fidelity: {history['val_fidelity'][-1]:.6f}")
    logger.info(f"  Epochs trained: {len(history['train_loss'])}")
    
    # Encode all states to latent space
    if not args.skip_encoding:
        logger.info("Encoding all states to latent space...")
        latent_reps = encode_all_states(model, groundstates, device)
        
        latent_file = save_latent_representations(latent_reps, output_dir, L)
        logger.info(f"Saved latent representations to {latent_file}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
