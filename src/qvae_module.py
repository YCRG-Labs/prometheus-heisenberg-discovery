"""Quantum-Aware Variational Autoencoder (Q-VAE) Module

This module implements a quantum-aware VAE with fidelity-based loss function
for learning latent representations of quantum ground states.

The Q-VAE architecture:
- Encoder: Maps wavefunctions to latent distributions (mu, logvar)
- Decoder: Reconstructs wavefunctions from latent samples
- Loss: Fidelity-based reconstruction + KL divergence regularization

Key features:
- Respects quantum mechanical structure (complex wavefunctions)
- Enforces wavefunction normalization in decoder
- Uses quantum fidelity F = |⟨ψ_in|ψ_recon⟩|² for reconstruction loss
- Separate models for each lattice size
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging

from src.exceptions import ComputationError, ValidationError

logger = logging.getLogger(__name__)


class QVAEEncoder(nn.Module):
    """Q-VAE Encoder: Maps wavefunctions to latent distributions
    
    Architecture:
    - Fully-connected layers with LayerNorm and ReLU
    - Outputs mu (mean) and logvar (log variance) for diagonal Gaussian
    
    The encoder learns q_φ(z|x) = N(z; μ(x), diag(σ²(x)))
    """
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        """Initialize QVAEEncoder
        
        Args:
            input_dim: Dimension of input (2 * Hilbert space dim for real/imag parts)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder layers
        self.layers = self._build_layers(input_dim, hidden_dims)
        
        # Output layers for mu and logvar
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        logger.debug(
            f"Initialized QVAEEncoder: input_dim={input_dim}, "
            f"latent_dim={latent_dim}, hidden_dims={hidden_dims}"
        )
    
    def _build_layers(self, input_dim: int, hidden_dims: List[int]) -> nn.ModuleList:
        """Build fully-connected layers with LayerNorm and ReLU
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            
        Returns:
            ModuleList of encoder layers
        """
        layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Layer normalization
            layers.append(nn.LayerNorm(hidden_dim))
            # ReLU activation
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        return layers
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Contains [Re(ψ), Im(ψ)] concatenated
        
        Returns:
            Tuple of (mu, logvar) tensors, each of shape (batch_size, latent_dim)
            - mu: Mean of latent distribution
            - logvar: Log variance of latent distribution
        """
        # Pass through encoder layers
        h = x
        for layer in self.layers:
            h = layer(h)
        
        # Compute mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class QVAEDecoder(nn.Module):
    """Q-VAE Decoder: Reconstructs wavefunctions from latent vectors
    
    Architecture:
    - Symmetric to encoder: fully-connected layers with LayerNorm and ReLU
    - Final layer: linear (no activation)
    - Post-processing: normalize to enforce ⟨ψ|ψ⟩ = 1
    """
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int]):
        """Initialize QVAEDecoder
        
        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output (2 * Hilbert space dim for real/imag parts)
            hidden_dims: List of hidden layer dimensions (e.g., [128, 256, 512])
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build decoder layers
        self.layers = self._build_layers(latent_dim, hidden_dims, output_dim)
        
        logger.debug(
            f"Initialized QVAEDecoder: latent_dim={latent_dim}, "
            f"output_dim={output_dim}, hidden_dims={hidden_dims}"
        )
    
    def _build_layers(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ) -> nn.ModuleList:
        """Build fully-connected layers with LayerNorm and ReLU
        
        Args:
            latent_dim: Latent dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            
        Returns:
            ModuleList of decoder layers
        """
        layers = nn.ModuleList()
        
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Layer normalization
            layers.append(nn.LayerNorm(hidden_dim))
            # ReLU activation
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Final output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return layers
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
        
        Returns:
            Reconstructed wavefunction tensor of shape (batch_size, output_dim)
            Contains [Re(ψ_recon), Im(ψ_recon)] concatenated and normalized
        """
        # Pass through decoder layers
        h = z
        for layer in self.layers:
            h = layer(h)
        
        # Normalize wavefunction to enforce ⟨ψ|ψ⟩ = 1
        psi_recon = self.normalize_wavefunction(h)
        
        return psi_recon
    
    def normalize_wavefunction(self, psi_recon: torch.Tensor) -> torch.Tensor:
        """Enforce normalization constraint ⟨ψ|ψ⟩ = 1
        
        The wavefunction is represented as [Re(ψ), Im(ψ)].
        Normalization: ⟨ψ|ψ⟩ = Σᵢ |ψᵢ|² = Σᵢ (Re(ψᵢ)² + Im(ψᵢ)²) = 1
        
        Args:
            psi_recon: Unnormalized wavefunction tensor (batch_size, output_dim)
        
        Returns:
            Normalized wavefunction tensor (batch_size, output_dim)
        """
        # Split into real and imaginary parts
        dim = psi_recon.shape[1] // 2
        re_psi = psi_recon[:, :dim]
        im_psi = psi_recon[:, dim:]
        
        # Compute norm: ||ψ||² = Σᵢ (Re(ψᵢ)² + Im(ψᵢ)²)
        norm_sq = torch.sum(re_psi**2 + im_psi**2, dim=1, keepdim=True)
        norm = torch.sqrt(norm_sq + 1e-10)  # Add small epsilon for numerical stability
        
        # Normalize: ψ_normalized = ψ / ||ψ||
        psi_normalized = psi_recon / norm
        
        return psi_normalized


class QVAE(nn.Module):
    """Quantum-Aware Variational Autoencoder
    
    Complete Q-VAE model combining encoder and decoder with quantum fidelity loss.
    
    Loss function:
    L = E[1 - F(ψ_in, ψ_recon)] + β * D_KL(q(z|x) || p(z))
    
    where:
    - F(ψ_in, ψ_recon) = |⟨ψ_in|ψ_recon⟩|² is quantum fidelity
    - D_KL is Kullback-Leibler divergence
    - β is regularization weight
    """
    
    def __init__(self, config: Any, hilbert_dim: int):
        """Initialize QVAE
        
        Args:
            config: Configuration object with qvae_architecture and training parameters
            hilbert_dim: Dimension of Hilbert space (basis dimension)
        """
        super().__init__()
        
        self.hilbert_dim = hilbert_dim
        self.input_dim = 2 * hilbert_dim  # Real and imaginary parts
        self.latent_dim = config.qvae_architecture.latent_dim
        self.beta = config.training.beta
        
        # Initialize encoder and decoder
        self.encoder = QVAEEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=config.qvae_architecture.encoder_layers
        )
        
        self.decoder = QVAEDecoder(
            latent_dim=self.latent_dim,
            output_dim=self.input_dim,
            hidden_dims=config.qvae_architecture.decoder_layers
        )
        
        logger.info(
            f"Initialized QVAE: hilbert_dim={hilbert_dim}, "
            f"latent_dim={self.latent_dim}, beta={self.beta}"
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution
        
        Instead of sampling z ~ N(μ, σ²), we sample ε ~ N(0, I) and compute:
        z = μ + σ ⊙ ε
        
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean tensor of shape (batch_size, latent_dim)
            logvar: Log variance tensor of shape (batch_size, latent_dim)
        
        Returns:
            Sampled latent vector z of shape (batch_size, latent_dim)
        """
        # Compute standard deviation from log variance
        std = torch.exp(0.5 * logvar)
        
        # Sample epsilon from standard normal
        eps = torch.randn_like(std)
        
        # Reparameterization: z = μ + σ * ε
        z = mu + std * eps
        
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through Q-VAE
        
        Args:
            x: Input wavefunction tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (reconstruction, mu, logvar)
            - reconstruction: Reconstructed wavefunction (batch_size, input_dim)
            - mu: Latent mean (batch_size, latent_dim)
            - logvar: Latent log variance (batch_size, latent_dim)
        """
        # Encode to latent distribution
        mu, logvar = self.encoder(x)
        
        # Sample from latent distribution using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruct wavefunction
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def compute_fidelity(
        self,
        psi_in: torch.Tensor,
        psi_recon: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantum fidelity F = |⟨ψ_in|ψ_recon⟩|²
        
        For complex wavefunctions represented as [Re(ψ), Im(ψ)]:
        ⟨ψ_in|ψ_recon⟩ = Σᵢ (Re(ψ_in)ᵢ - i*Im(ψ_in)ᵢ) * (Re(ψ_recon)ᵢ + i*Im(ψ_recon)ᵢ)
                       = Σᵢ (Re(ψ_in)ᵢ*Re(ψ_recon)ᵢ + Im(ψ_in)ᵢ*Im(ψ_recon)ᵢ)
                         + i*Σᵢ (Re(ψ_in)ᵢ*Im(ψ_recon)ᵢ - Im(ψ_in)ᵢ*Re(ψ_recon)ᵢ)
        
        Fidelity F = |⟨ψ_in|ψ_recon⟩|² = (Re(overlap))² + (Im(overlap))²
        
        Args:
            psi_in: Input wavefunction (batch_size, input_dim)
            psi_recon: Reconstructed wavefunction (batch_size, input_dim)
        
        Returns:
            Fidelity tensor of shape (batch_size,) with values in [0, 1]
        """
        # Split into real and imaginary parts
        dim = psi_in.shape[1] // 2
        re_in = psi_in[:, :dim]
        im_in = psi_in[:, dim:]
        re_recon = psi_recon[:, :dim]
        im_recon = psi_recon[:, dim:]
        
        # Compute complex inner product ⟨ψ_in|ψ_recon⟩
        # Real part: Σᵢ (Re(ψ_in)ᵢ*Re(ψ_recon)ᵢ + Im(ψ_in)ᵢ*Im(ψ_recon)ᵢ)
        overlap_re = torch.sum(re_in * re_recon + im_in * im_recon, dim=1)
        
        # Imaginary part: Σᵢ (Re(ψ_in)ᵢ*Im(ψ_recon)ᵢ - Im(ψ_in)ᵢ*Re(ψ_recon)ᵢ)
        overlap_im = torch.sum(re_in * im_recon - im_in * re_recon, dim=1)
        
        # Fidelity: F = |overlap|² = (Re(overlap))² + (Im(overlap))²
        fidelity = overlap_re**2 + overlap_im**2
        
        # Clamp to [0, 1] for numerical stability
        fidelity = torch.clamp(fidelity, 0.0, 1.0)
        
        return fidelity
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss with fidelity-based reconstruction
        
        Loss components:
        1. Fidelity loss: L_fid = 1 - F(ψ_in, ψ_recon)
        2. KL divergence: D_KL = -0.5 * Σ(1 + logvar - μ² - exp(logvar))
        
        Total loss: L = E[L_fid] + β * D_KL
        
        Args:
            x: Input wavefunction (batch_size, input_dim)
            recon: Reconstructed wavefunction (batch_size, input_dim)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log variance (batch_size, latent_dim)
        
        Returns:
            Dictionary with keys:
            - 'loss': Total loss (scalar)
            - 'fidelity_loss': Fidelity loss component (scalar)
            - 'kl_loss': KL divergence component (scalar)
            - 'fidelity': Average fidelity (scalar, for monitoring)
            
        Raises:
            ComputationError: If loss contains NaN or Inf values
        """
        # Check for NaN or Inf in inputs
        if not torch.all(torch.isfinite(x)):
            raise ComputationError(
                "Input wavefunction contains NaN or Inf",
                context={'operation': 'loss_function', 'tensor': 'x'}
            )
        if not torch.all(torch.isfinite(recon)):
            raise ComputationError(
                "Reconstructed wavefunction contains NaN or Inf",
                context={'operation': 'loss_function', 'tensor': 'recon'}
            )
        if not torch.all(torch.isfinite(mu)):
            raise ComputationError(
                "Latent mean contains NaN or Inf",
                context={'operation': 'loss_function', 'tensor': 'mu'}
            )
        if not torch.all(torch.isfinite(logvar)):
            raise ComputationError(
                "Latent logvar contains NaN or Inf",
                context={'operation': 'loss_function', 'tensor': 'logvar'}
            )
        
        # Compute fidelity for each sample in batch
        fidelity = self.compute_fidelity(x, recon)
        
        # Check fidelity is valid
        if not torch.all(torch.isfinite(fidelity)):
            raise ComputationError(
                "Fidelity computation produced NaN or Inf",
                context={'operation': 'compute_fidelity'}
            )
        
        # Fidelity loss: 1 - F (we want to maximize F, so minimize 1-F)
        fidelity_loss = torch.mean(1.0 - fidelity)
        
        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # D_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Check for NaN in loss components
        if not torch.isfinite(fidelity_loss):
            raise ComputationError(
                "Fidelity loss is NaN or Inf",
                context={'operation': 'loss_function', 'fidelity_loss': float(fidelity_loss)}
            )
        if not torch.isfinite(kl_loss):
            raise ComputationError(
                "KL loss is NaN or Inf",
                context={'operation': 'loss_function', 'kl_loss': float(kl_loss)}
            )
        
        # Total loss: fidelity loss + β * KL divergence
        total_loss = fidelity_loss + self.beta * kl_loss
        
        # Final check on total loss
        if not torch.isfinite(total_loss):
            raise ComputationError(
                "Total loss is NaN or Inf",
                context={
                    'operation': 'loss_function',
                    'total_loss': float(total_loss),
                    'fidelity_loss': float(fidelity_loss),
                    'kl_loss': float(kl_loss),
                    'beta': self.beta
                }
            )
        
        return {
            'loss': total_loss,
            'fidelity_loss': fidelity_loss,
            'kl_loss': kl_loss,
            'fidelity': torch.mean(fidelity)  # For monitoring
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (deterministic, returns mean)
        
        This method is used for inference to get deterministic latent representations.
        It returns the mean μ of the latent distribution without sampling.
        
        Args:
            x: Input wavefunction tensor (batch_size, input_dim)
        
        Returns:
            Latent mean tensor (batch_size, latent_dim)
        """
        mu, _ = self.encoder(x)
        return mu



class QVAETrainer:
    """Trainer for Q-VAE with early stopping and learning rate scheduling
    
    Implements the training loop with:
    - Gradient clipping for stability
    - Early stopping based on validation loss
    - Cosine annealing learning rate schedule
    - Data augmentation using Sz symmetry
    - Comprehensive logging of training metrics
    """
    
    def __init__(self, model: QVAE, config: Any):
        """Initialize QVAETrainer
        
        Args:
            model: QVAE model to train
            config: Configuration object with training parameters
        """
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.max_epochs
        )
        
        # Training parameters
        self.max_epochs = config.training.max_epochs
        self.patience = config.training.patience
        self.gradient_clip = config.training.gradient_clip
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
        logger.info(
            f"Initialized QVAETrainer: lr={config.training.learning_rate}, "
            f"max_epochs={self.max_epochs}, patience={self.patience}"
        )
    
    def apply_data_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation using Sz symmetry
        
        The J1-J2 Heisenberg Hamiltonian has Sz symmetry: flipping all spins
        (σ → -σ) leaves the physics unchanged. We exploit this to augment
        the training data.
        
        For a wavefunction represented as [Re(ψ), Im(ψ)], the Sz flip
        corresponds to complex conjugation: ψ → ψ*
        This maps [Re(ψ), Im(ψ)] → [Re(ψ), -Im(ψ)]
        
        Args:
            x: Input tensor (batch_size, input_dim) with [Re(ψ), Im(ψ)]
        
        Returns:
            Augmented tensor with 50% probability of Sz flip applied
        """
        # Apply augmentation with 50% probability
        if torch.rand(1).item() < 0.5:
            # Split into real and imaginary parts
            dim = x.shape[1] // 2
            re_part = x[:, :dim]
            im_part = x[:, dim:]
            
            # Apply Sz flip: ψ → ψ* means [Re(ψ), Im(ψ)] → [Re(ψ), -Im(ψ)]
            x_augmented = torch.cat([re_part, -im_part], dim=1)
            return x_augmented
        else:
            return x
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Dictionary with average losses:
            - 'loss': Total loss
            - 'fidelity_loss': Fidelity loss component
            - 'kl_loss': KL divergence component
            - 'fidelity': Average fidelity
        """
        self.model.train()
        
        epoch_losses = {
            'loss': 0.0,
            'fidelity_loss': 0.0,
            'kl_loss': 0.0,
            'fidelity': 0.0
        }
        n_batches = 0
        
        for batch_data in train_loader:
            # Get batch (handle both tuple and tensor formats)
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0]
            else:
                x = batch_data
            
            # Apply data augmentation
            x = self.apply_data_augmentation(x)
            
            # Forward pass
            recon, mu, logvar = self.model(x)
            
            # Compute loss
            loss_dict = self.model.loss_function(x, recon, mu, logvar)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
            n_batches += 1
        
        # Average losses over batches
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary with average losses (same keys as train_epoch)
        """
        self.model.eval()
        
        val_losses = {
            'loss': 0.0,
            'fidelity_loss': 0.0,
            'kl_loss': 0.0,
            'fidelity': 0.0
        }
        n_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Get batch (handle both tuple and tensor formats)
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data
                
                # Forward pass
                recon, mu, logvar = self.model(x)
                
                # Compute loss
                loss_dict = self.model.loss_function(x, recon, mu, logvar)
                
                # Accumulate losses
                for key in val_losses:
                    val_losses[key] += loss_dict[key].item()
                n_batches += 1
        
        # Average losses over batches
        for key in val_losses:
            val_losses[key] /= n_batches
        
        return val_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping
        
        Trains the model until:
        - Maximum epochs reached, OR
        - Early stopping triggered (no improvement for patience epochs)
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary with training history:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'train_fidelity': List of training fidelities per epoch
            - 'val_fidelity': List of validation fidelities per epoch
            - 'learning_rate': List of learning rates per epoch
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_fidelity': [],
            'val_fidelity': [],
            'learning_rate': []
        }
        
        logger.info("Starting Q-VAE training...")
        
        # Check if using GPU
        device = next(self.model.parameters()).device
        using_gpu = device.type == 'cuda'
        
        for epoch in range(self.max_epochs):
            # Train for one epoch
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_losses['loss'])
            history['val_loss'].append(val_losses['loss'])
            history['train_fidelity'].append(train_losses['fidelity'])
            history['val_fidelity'].append(val_losses['fidelity'])
            history['learning_rate'].append(current_lr)
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = (
                    f"Epoch {epoch+1}/{self.max_epochs}: "
                    f"train_loss={train_losses['loss']:.6f}, "
                    f"val_loss={val_losses['loss']:.6f}, "
                    f"train_fid={train_losses['fidelity']:.6f}, "
                    f"val_fid={val_losses['fidelity']:.6f}, "
                    f"lr={current_lr:.6e}"
                )
                
                # Add GPU memory info if using GPU
                if using_gpu:
                    gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                    gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    log_msg += f", GPU_mem={gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f} GB"
                
                logger.info(log_msg)
            
            # Early stopping check
            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.epochs_without_improvement = 0
                # Save best model state
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                logger.debug(f"New best validation loss: {self.best_val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"No improvement for {self.patience} epochs."
                    )
                    break
            
            # Learning rate scheduler step
            self.scheduler.step()
            
            # Clear GPU cache periodically
            if using_gpu and (epoch + 1) % 50 == 0:
                torch.cuda.empty_cache()
                logger.debug("Cleared GPU cache")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val_loss={self.best_val_loss:.6f}")
        
        # Final GPU cache clear
        if using_gpu:
            torch.cuda.empty_cache()
        
        logger.info("Training completed.")
        
        return history



class QVAEModule:
    """High-level module for Q-VAE training and encoding
    
    Manages Q-VAE models for all lattice sizes, handling:
    - Dataset preparation with train/validation split
    - Training separate models for each lattice size
    - Encoding all ground states to latent representations
    - Model checkpointing and loading
    """
    
    def __init__(self, config: Any):
        """Initialize QVAEModule
        
        Args:
            config: Configuration object with Q-VAE and training parameters
        """
        self.config = config
        self.models = {}  # Dictionary mapping L -> QVAE model
        self.trainers = {}  # Dictionary mapping L -> QVAETrainer
        self.training_histories = {}  # Dictionary mapping L -> training history
        
        logger.info("Initialized QVAEModule")
    
    def prepare_dataset(
        self,
        states: Dict[Tuple[float, int], Any],
        L: int,
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train/validation dataloaders for given lattice size
        
        Filters states for the specified lattice size, converts to real vectors,
        and creates train/validation split.
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
            L: Lattice size to prepare dataset for
            train_split: Fraction of data for training (default: 0.8)
        
        Returns:
            Tuple of (train_loader, val_loader)
        
        Raises:
            ValueError: If no states found for lattice size L
        """
        # Filter states for this lattice size
        states_L = {
            (j2_j1, l): state
            for (j2_j1, l), state in states.items()
            if l == L
        }
        
        if not states_L:
            raise ValueError(f"No states found for lattice size L={L}")
        
        logger.info(f"Preparing dataset for L={L}: {len(states_L)} states")
        
        # Convert ground states to real vectors
        data_list = []
        for (j2_j1, _), state in sorted(states_L.items()):
            real_vector = state.to_real_vector()
            data_list.append(real_vector)
        
        # Convert to tensor
        data_tensor = torch.tensor(np.array(data_list), dtype=torch.float32)
        
        # Create dataset
        dataset = TensorDataset(data_tensor)
        
        # Split into train and validation
        n_total = len(dataset)
        n_train = int(train_split * n_total)
        n_val = n_total - n_train
        
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create dataloaders
        batch_size = min(self.config.training.batch_size, n_train)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        logger.info(
            f"Dataset prepared: {n_train} train, {n_val} val, "
            f"batch_size={batch_size}"
        )
        
        return train_loader, val_loader
    
    def train_for_lattice_size(
        self,
        states: Dict[Tuple[float, int], Any],
        L: int,
        device: Optional[torch.device] = None
    ) -> QVAE:
        """Train Q-VAE for specific lattice size
        
        Creates a new Q-VAE model, prepares the dataset, and trains the model
        with early stopping and learning rate scheduling.
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
            L: Lattice size to train for
            device: Device to train on (default: from config or auto-detect)
        
        Returns:
            Trained QVAE model
        
        Raises:
            ValueError: If no states found for lattice size L
        """
        logger.info(f"Training Q-VAE for lattice size L={L}")
        
        # Get device from config if not specified
        if device is None:
            device = self.config.get_device()
        
        logger.info(f"Using device: {device}")
        
        # Log GPU information if using CUDA
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
            
            # Log initial GPU memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.4f} GB")
                logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved(device) / 1e9:.4f} GB")
        
        # Get a sample state to determine Hilbert space dimension
        sample_state = next(
            state for (j2_j1, l), state in states.items() if l == L
        )
        hilbert_dim = len(sample_state.coefficients)
        
        logger.info(f"Hilbert space dimension for L={L}: {hilbert_dim}")
        
        # Create Q-VAE model
        model = QVAE(self.config, hilbert_dim)
        model = model.to(device)
        
        # Prepare dataset
        train_loader, val_loader = self.prepare_dataset(states, L)
        
        # Move data to device
        train_loader = self._move_loader_to_device(train_loader, device)
        val_loader = self._move_loader_to_device(val_loader, device)
        
        # Create trainer
        trainer = QVAETrainer(model, self.config)
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        # Store model, trainer, and history
        self.models[L] = model
        self.trainers[L] = trainer
        self.training_histories[L] = history
        
        logger.info(f"Training completed for L={L}")
        
        return model
    
    def _move_loader_to_device(
        self,
        loader: DataLoader,
        device: torch.device
    ) -> DataLoader:
        """Move dataloader tensors to specified device
        
        Args:
            loader: DataLoader to move
            device: Target device
        
        Returns:
            New DataLoader with data on target device
        """
        # Extract dataset from loader
        dataset = loader.dataset
        
        # Get the underlying dataset if it's a Subset
        if hasattr(dataset, 'dataset'):
            # It's a Subset from random_split
            base_dataset = dataset.dataset
            indices = dataset.indices
            
            # Get tensors from base dataset
            tensors = base_dataset.tensors
            
            # Move tensors to device
            device_tensors = tuple(t.to(device) for t in tensors)
            
            # Create new dataset on device
            device_base_dataset = TensorDataset(*device_tensors)
            
            # Create subset with same indices
            from torch.utils.data import Subset
            device_dataset = Subset(device_base_dataset, indices)
        else:
            # It's a regular TensorDataset
            tensors = dataset.tensors
            device_tensors = tuple(t.to(device) for t in tensors)
            device_dataset = TensorDataset(*device_tensors)
        
        # Create new dataloader
        return DataLoader(
            device_dataset,
            batch_size=loader.batch_size,
            shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
            drop_last=loader.drop_last
        )
    
    def train_all(
        self,
        states: Dict[Tuple[float, int], Any],
        device: Optional[torch.device] = None
    ) -> None:
        """Train Q-VAE for all lattice sizes
        
        Trains separate Q-VAE models for each lattice size present in the
        states dictionary.
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
            device: Device to train on (default: from config or auto-detect)
        """
        # Get device from config if not specified
        if device is None:
            device = self.config.get_device()
        
        # Get unique lattice sizes from states
        lattice_sizes = sorted(set(L for (_, L) in states.keys()))
        
        logger.info(f"Training Q-VAE for lattice sizes: {lattice_sizes}")
        logger.info(f"Using device: {device}")
        
        for L in lattice_sizes:
            self.train_for_lattice_size(states, L, device)
        
        logger.info("Training completed for all lattice sizes")
    
    def encode_all(
        self,
        states: Dict[Tuple[float, int], Any],
        device: Optional[torch.device] = None
    ) -> Dict[Tuple[float, int], np.ndarray]:
        """Encode all states to latent representations
        
        Uses the trained Q-VAE models to encode each ground state to its
        latent representation (mean of the latent distribution).
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
            device: Device to use for encoding (default: from config or auto-detect)
        
        Returns:
            Dictionary mapping (j2_j1, L) -> latent representation (numpy array)
        
        Raises:
            RuntimeError: If model not trained for a required lattice size
        """
        logger.info("Encoding all states to latent representations")
        
        # Get device from config if not specified
        if device is None:
            device = self.config.get_device()
        
        latent_reps = {}
        
        for (j2_j1, L), state in sorted(states.items()):
            # Check if model exists for this lattice size
            if L not in self.models:
                raise RuntimeError(
                    f"No trained model found for lattice size L={L}. "
                    f"Train model first using train_for_lattice_size() or train_all()."
                )
            
            # Get model
            model = self.models[L]
            model.eval()
            
            # Convert state to real vector
            real_vector = state.to_real_vector()
            x = torch.tensor(real_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dim
            x = x.to(device)
            
            # Encode (deterministic, returns mean)
            with torch.no_grad():
                z = model.encode(x)
            
            # Convert to numpy and remove batch dimension
            z_np = z.cpu().numpy().squeeze(0)
            
            latent_reps[(j2_j1, L)] = z_np
        
        logger.info(f"Encoded {len(latent_reps)} states")
        
        return latent_reps
    
    def save_models(self, checkpoint_dir: str) -> None:
        """Save all trained models to checkpoint directory
        
        Args:
            checkpoint_dir: Directory to save model checkpoints
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        for L, model in self.models.items():
            model_path = checkpoint_path / f"qvae_L{L}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config,
                'L': L,
                'hilbert_dim': model.hilbert_dim,
                'training_history': self.training_histories.get(L, {})
            }, model_path)
            logger.info(f"Saved model for L={L} to {model_path}")
    
    def load_models(self, checkpoint_dir: str, device: Optional[torch.device] = None) -> None:
        """Load trained models from checkpoint directory
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            device: Device to load models to (default: from config or auto-detect)
        """
        # Get device from config if not specified
        if device is None:
            device = self.config.get_device()
        
        checkpoint_path = Path(checkpoint_dir)
        
        # Find all model checkpoint files
        model_files = list(checkpoint_path.glob("qvae_L*.pt"))
        
        if not model_files:
            raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")
        
        for model_file in model_files:
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location=device)
            
            L = checkpoint['L']
            hilbert_dim = checkpoint['hilbert_dim']
            
            # Create model
            model = QVAE(self.config, hilbert_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            # Store model and history
            self.models[L] = model
            self.training_histories[L] = checkpoint.get('training_history', {})
            
            logger.info(f"Loaded model for L={L} from {model_file}")
        
        logger.info(f"Loaded {len(self.models)} models")

    
    def save_model(self, L: int, storage: Any) -> None:
        """Save a single model using DataStorage
        
        Args:
            L: Lattice size
            storage: DataStorage instance
        """
        if L not in self.models:
            raise ValueError(f"No model found for L={L}")
        
        model = self.models[L]
        metadata = {
            'hilbert_dim': model.hilbert_dim,
            'latent_dim': model.latent_dim,
            'training_history': self.training_histories.get(L, {})
        }
        
        storage.save_qvae_model(model, L, metadata)
    
    def load_model(self, L: int, storage: Any, device: Optional[torch.device] = None) -> None:
        """Load a single model using DataStorage
        
        Args:
            L: Lattice size
            storage: DataStorage instance
            device: Device to load model to (default: from config or auto-detect)
        """
        # Get device from config if not specified
        if device is None:
            device = self.config.get_device()
        
        # Get sample state to determine Hilbert space dimension
        # This is a workaround - ideally we'd store hilbert_dim in metadata
        # For now, we'll create a model with the right dimensions
        # The hilbert_dim should be stored in the checkpoint metadata
        
        # Create a temporary model to load the checkpoint
        # We need to know the hilbert_dim, which should be in the metadata
        metadata = storage.load_qvae_model(None, L)  # Load metadata first
        
        if 'hilbert_dim' in metadata:
            hilbert_dim = metadata['hilbert_dim']
        else:
            # Fallback: compute from lattice size
            # For Sz=0 sector: dim = C(N, N/2) where N = L*L
            from scipy.special import comb
            N = L * L
            hilbert_dim = int(comb(N, N // 2))
        
        # Create model
        model = QVAE(self.config, hilbert_dim)
        
        # Load state dict
        storage.load_qvae_model(model, L)
        
        model = model.to(device)
        model.eval()
        
        # Store model
        self.models[L] = model
        
        # Load training history if available
        if 'training_history' in metadata:
            self.training_histories[L] = metadata['training_history']
