"""Unit tests for Q-VAE module"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qvae_module import QVAEEncoder, QVAEDecoder, QVAE
from config import Config


class TestQVAEEncoder:
    """Test QVAEEncoder class"""
    
    def test_encoder_initialization(self):
        """Test encoder initializes correctly"""
        input_dim = 100
        latent_dim = 8
        hidden_dims = [64, 32]
        
        encoder = QVAEEncoder(input_dim, latent_dim, hidden_dims)
        
        assert encoder.input_dim == input_dim
        assert encoder.latent_dim == latent_dim
        assert encoder.hidden_dims == hidden_dims
    
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        input_dim = 100
        latent_dim = 8
        hidden_dims = [64, 32]
        batch_size = 4
        
        encoder = QVAEEncoder(input_dim, latent_dim, hidden_dims)
        
        # Create random input
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        mu, logvar = encoder(x)
        
        # Check output shapes
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        
        # Check outputs are finite
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))


class TestQVAEDecoder:
    """Test QVAEDecoder class"""
    
    def test_decoder_initialization(self):
        """Test decoder initializes correctly"""
        latent_dim = 8
        output_dim = 100
        hidden_dims = [32, 64]
        
        decoder = QVAEDecoder(latent_dim, output_dim, hidden_dims)
        
        assert decoder.latent_dim == latent_dim
        assert decoder.output_dim == output_dim
        assert decoder.hidden_dims == hidden_dims
    
    def test_decoder_forward(self):
        """Test decoder forward pass"""
        latent_dim = 8
        output_dim = 100
        hidden_dims = [32, 64]
        batch_size = 4
        
        decoder = QVAEDecoder(latent_dim, output_dim, hidden_dims)
        
        # Create random latent vector
        z = torch.randn(batch_size, latent_dim)
        
        # Forward pass
        recon = decoder(z)
        
        # Check output shape
        assert recon.shape == (batch_size, output_dim)
        
        # Check output is finite
        assert torch.all(torch.isfinite(recon))
    
    def test_normalize_wavefunction(self):
        """Test wavefunction normalization"""
        latent_dim = 8
        output_dim = 100  # Must be even (real + imag parts)
        hidden_dims = [32, 64]
        batch_size = 4
        
        decoder = QVAEDecoder(latent_dim, output_dim, hidden_dims)
        
        # Create unnormalized wavefunction
        psi_unnorm = torch.randn(batch_size, output_dim) * 10.0
        
        # Normalize
        psi_norm = decoder.normalize_wavefunction(psi_unnorm)
        
        # Check normalization: ||ψ||² = Σᵢ (Re(ψᵢ)² + Im(ψᵢ)²) = 1
        dim = output_dim // 2
        re_psi = psi_norm[:, :dim]
        im_psi = psi_norm[:, dim:]
        norm_sq = torch.sum(re_psi**2 + im_psi**2, dim=1)
        
        # Check each sample is normalized
        assert torch.allclose(norm_sq, torch.ones(batch_size), atol=1e-6)


class TestQVAE:
    """Test QVAE class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config.from_yaml("configs/default_config.yaml")
    
    def test_qvae_initialization(self, config):
        """Test QVAE initializes correctly"""
        hilbert_dim = 50
        
        qvae = QVAE(config, hilbert_dim)
        
        assert qvae.hilbert_dim == hilbert_dim
        assert qvae.input_dim == 2 * hilbert_dim
        assert qvae.latent_dim == config.qvae_architecture.latent_dim
        assert qvae.beta == config.training.beta
    
    def test_qvae_forward(self, config):
        """Test QVAE forward pass"""
        hilbert_dim = 50
        batch_size = 4
        
        qvae = QVAE(config, hilbert_dim)
        
        # Create random input wavefunction
        x = torch.randn(batch_size, 2 * hilbert_dim)
        
        # Normalize input
        dim = hilbert_dim
        re_x = x[:, :dim]
        im_x = x[:, dim:]
        norm = torch.sqrt(torch.sum(re_x**2 + im_x**2, dim=1, keepdim=True))
        x = x / norm
        
        # Forward pass
        recon, mu, logvar = qvae(x)
        
        # Check output shapes
        assert recon.shape == (batch_size, 2 * hilbert_dim)
        assert mu.shape == (batch_size, config.qvae_architecture.latent_dim)
        assert logvar.shape == (batch_size, config.qvae_architecture.latent_dim)
        
        # Check outputs are finite
        assert torch.all(torch.isfinite(recon))
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))
    
    def test_reparameterize(self, config):
        """Test reparameterization trick"""
        hilbert_dim = 50
        batch_size = 4
        latent_dim = config.qvae_architecture.latent_dim
        
        qvae = QVAE(config, hilbert_dim)
        
        # Create mu and logvar
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        
        # Sample using reparameterization
        z = qvae.reparameterize(mu, logvar)
        
        # Check output shape
        assert z.shape == (batch_size, latent_dim)
        
        # Check output is finite
        assert torch.all(torch.isfinite(z))
    
    def test_compute_fidelity(self, config):
        """Test fidelity computation"""
        hilbert_dim = 50
        batch_size = 4
        
        qvae = QVAE(config, hilbert_dim)
        
        # Create two normalized wavefunctions
        psi1 = torch.randn(batch_size, 2 * hilbert_dim)
        psi2 = torch.randn(batch_size, 2 * hilbert_dim)
        
        # Normalize both
        for psi in [psi1, psi2]:
            dim = hilbert_dim
            re_psi = psi[:, :dim]
            im_psi = psi[:, dim:]
            norm = torch.sqrt(torch.sum(re_psi**2 + im_psi**2, dim=1, keepdim=True))
            psi.data = psi / norm
        
        # Compute fidelity
        fidelity = qvae.compute_fidelity(psi1, psi2)
        
        # Check output shape
        assert fidelity.shape == (batch_size,)
        
        # Check fidelity is in [0, 1]
        assert torch.all(fidelity >= 0.0)
        assert torch.all(fidelity <= 1.0)
        
        # Check fidelity of identical states is 1
        fidelity_self = qvae.compute_fidelity(psi1, psi1)
        assert torch.allclose(fidelity_self, torch.ones(batch_size), atol=1e-5)
    
    def test_loss_function(self, config):
        """Test loss function computation"""
        hilbert_dim = 50
        batch_size = 4
        
        qvae = QVAE(config, hilbert_dim)
        
        # Create normalized input wavefunction
        x = torch.randn(batch_size, 2 * hilbert_dim)
        dim = hilbert_dim
        re_x = x[:, :dim]
        im_x = x[:, dim:]
        norm = torch.sqrt(torch.sum(re_x**2 + im_x**2, dim=1, keepdim=True))
        x = x / norm
        
        # Forward pass
        recon, mu, logvar = qvae(x)
        
        # Compute loss
        loss_dict = qvae.loss_function(x, recon, mu, logvar)
        
        # Check loss components exist
        assert 'loss' in loss_dict
        assert 'fidelity_loss' in loss_dict
        assert 'kl_loss' in loss_dict
        assert 'fidelity' in loss_dict
        
        # Check all losses are finite
        for key, value in loss_dict.items():
            assert torch.isfinite(value), f"{key} is not finite"
        
        # Check fidelity is in [0, 1]
        assert loss_dict['fidelity'] >= 0.0
        assert loss_dict['fidelity'] <= 1.0
        
        # Check KL divergence is non-negative
        assert loss_dict['kl_loss'] >= 0.0
    
    def test_encode(self, config):
        """Test deterministic encoding"""
        hilbert_dim = 50
        batch_size = 4
        
        qvae = QVAE(config, hilbert_dim)
        
        # Create normalized input wavefunction
        x = torch.randn(batch_size, 2 * hilbert_dim)
        dim = hilbert_dim
        re_x = x[:, :dim]
        im_x = x[:, dim:]
        norm = torch.sqrt(torch.sum(re_x**2 + im_x**2, dim=1, keepdim=True))
        x = x / norm
        
        # Encode
        z = qvae.encode(x)
        
        # Check output shape
        assert z.shape == (batch_size, config.qvae_architecture.latent_dim)
        
        # Check output is finite
        assert torch.all(torch.isfinite(z))
        
        # Check deterministic: encoding same input twice gives same result
        z2 = qvae.encode(x)
        assert torch.allclose(z, z2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
