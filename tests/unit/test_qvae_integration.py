"""Integration tests for Q-VAE with ED module"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qvae_module import QVAE
from ed_module import J1J2Hamiltonian, GroundState
from config import Config


class TestQVAEIntegration:
    """Test Q-VAE integration with ED module"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config.from_yaml("configs/default_config.yaml")
    
    @pytest.fixture
    def ground_state(self):
        """Create a test ground state"""
        L = 4
        j2_j1 = 0.5
        
        # Compute ground state
        ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        return state
    
    def test_qvae_with_ground_state(self, config, ground_state):
        """Test Q-VAE can process ground state wavefunctions"""
        # Get Hilbert space dimension from ground state
        hilbert_dim = len(ground_state.coefficients)
        
        # Create Q-VAE
        qvae = QVAE(config, hilbert_dim)
        
        # Convert ground state to real vector
        x = ground_state.to_real_vector()
        
        # Convert to torch tensor and add batch dimension
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        
        # Forward pass
        recon, mu, logvar = qvae(x_tensor)
        
        # Check outputs
        assert recon.shape == (1, 2 * hilbert_dim)
        assert mu.shape == (1, config.qvae_architecture.latent_dim)
        assert logvar.shape == (1, config.qvae_architecture.latent_dim)
        
        # Check all outputs are finite
        assert torch.all(torch.isfinite(recon))
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))
        
        # Compute loss
        loss_dict = qvae.loss_function(x_tensor, recon, mu, logvar)
        
        # Check loss is finite
        assert torch.isfinite(loss_dict['loss'])
        
        # Check fidelity is reasonable (should be > 0 even for untrained model)
        assert loss_dict['fidelity'] > 0.0
        assert loss_dict['fidelity'] <= 1.0
    
    def test_qvae_batch_processing(self, config):
        """Test Q-VAE can process batch of ground states"""
        L = 4
        j2_j1_values = [0.3, 0.4, 0.5, 0.6]
        
        # Compute multiple ground states
        states = []
        for j2_j1 in j2_j1_values:
            ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
            ham.build_hamiltonian()
            state = ham.compute_ground_state()
            states.append(state)
        
        # Get Hilbert space dimension
        hilbert_dim = len(states[0].coefficients)
        
        # Create Q-VAE
        qvae = QVAE(config, hilbert_dim)
        
        # Convert all states to batch tensor
        x_list = [state.to_real_vector() for state in states]
        x_batch = torch.from_numpy(np.stack(x_list)).float()
        
        # Forward pass
        recon, mu, logvar = qvae(x_batch)
        
        # Check outputs
        batch_size = len(j2_j1_values)
        assert recon.shape == (batch_size, 2 * hilbert_dim)
        assert mu.shape == (batch_size, config.qvae_architecture.latent_dim)
        assert logvar.shape == (batch_size, config.qvae_architecture.latent_dim)
        
        # Check all outputs are finite
        assert torch.all(torch.isfinite(recon))
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))
        
        # Compute loss
        loss_dict = qvae.loss_function(x_batch, recon, mu, logvar)
        
        # Check loss is finite
        assert torch.isfinite(loss_dict['loss'])
    
    def test_qvae_encode_ground_states(self, config):
        """Test Q-VAE can encode ground states to latent space"""
        L = 4
        j2_j1_values = [0.3, 0.5, 0.7]
        
        # Compute ground states
        states = []
        for j2_j1 in j2_j1_values:
            ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
            ham.build_hamiltonian()
            state = ham.compute_ground_state()
            states.append(state)
        
        # Get Hilbert space dimension
        hilbert_dim = len(states[0].coefficients)
        
        # Create Q-VAE
        qvae = QVAE(config, hilbert_dim)
        
        # Convert states to batch tensor
        x_list = [state.to_real_vector() for state in states]
        x_batch = torch.from_numpy(np.stack(x_list)).float()
        
        # Encode to latent space
        z = qvae.encode(x_batch)
        
        # Check output shape
        assert z.shape == (len(j2_j1_values), config.qvae_architecture.latent_dim)
        
        # Check output is finite
        assert torch.all(torch.isfinite(z))
        
        # Check latent representations are different for different j2_j1
        # (even for untrained model, they should differ due to random initialization)
        z_np = z.detach().numpy()
        for i in range(len(j2_j1_values)):
            for j in range(i + 1, len(j2_j1_values)):
                # Check they're not identical
                assert not np.allclose(z_np[i], z_np[j])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
