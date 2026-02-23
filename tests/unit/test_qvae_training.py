"""Unit tests for Q-VAE training infrastructure"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qvae_module import QVAE, QVAETrainer, QVAEModule
from ed_module import J1J2Hamiltonian
from config import Config


class TestQVAETrainer:
    """Test QVAETrainer class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with reduced epochs for testing"""
        config = Config.from_yaml("configs/default_config.yaml")
        # Reduce epochs for faster testing
        config.training.max_epochs = 10
        config.training.patience = 5
        return config
    
    @pytest.fixture
    def model_and_data(self, config):
        """Create model and sample data"""
        hilbert_dim = 50
        batch_size = 8
        
        # Create model
        model = QVAE(config, hilbert_dim)
        
        # Create synthetic normalized wavefunctions
        data = []
        for _ in range(batch_size):
            x = torch.randn(2 * hilbert_dim)
            # Normalize
            dim = hilbert_dim
            re_x = x[:dim]
            im_x = x[dim:]
            norm = torch.sqrt(torch.sum(re_x**2 + im_x**2))
            x = x / norm
            data.append(x)
        
        data_tensor = torch.stack(data)
        
        return model, data_tensor
    
    def test_trainer_initialization(self, config, model_and_data):
        """Test trainer initializes correctly"""
        model, _ = model_and_data
        
        trainer = QVAETrainer(model, config)
        
        assert trainer.model is model
        assert trainer.config is config
        assert trainer.max_epochs == config.training.max_epochs
        assert trainer.patience == config.training.patience
        assert trainer.gradient_clip == config.training.gradient_clip
    
    def test_apply_data_augmentation(self, config, model_and_data):
        """Test data augmentation using Sz symmetry"""
        model, data = model_and_data
        trainer = QVAETrainer(model, config)
        
        # Apply augmentation multiple times
        augmented_same = []
        augmented_diff = []
        
        for _ in range(10):
            aug = trainer.apply_data_augmentation(data[0:1])
            
            # Check if it's the same or flipped
            if torch.allclose(aug, data[0:1]):
                augmented_same.append(True)
            else:
                # Check if imaginary part is flipped
                dim = data.shape[1] // 2
                re_orig = data[0:1, :dim]
                im_orig = data[0:1, dim:]
                re_aug = aug[:, :dim]
                im_aug = aug[:, dim:]
                
                if torch.allclose(re_aug, re_orig) and torch.allclose(im_aug, -im_orig):
                    augmented_diff.append(True)
        
        # Should have both augmented and non-augmented samples
        assert len(augmented_same) > 0 or len(augmented_diff) > 0
    
    def test_train_epoch(self, config, model_and_data):
        """Test training for one epoch"""
        model, data = model_and_data
        trainer = QVAETrainer(model, config)
        
        # Create dataloader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Train one epoch
        losses = trainer.train_epoch(loader)
        
        # Check loss dictionary
        assert 'loss' in losses
        assert 'fidelity_loss' in losses
        assert 'kl_loss' in losses
        assert 'fidelity' in losses
        
        # Check all losses are finite
        for key, value in losses.items():
            assert np.isfinite(value), f"{key} is not finite"
    
    def test_validate(self, config, model_and_data):
        """Test validation"""
        model, data = model_and_data
        trainer = QVAETrainer(model, config)
        
        # Create dataloader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Validate
        losses = trainer.validate(loader)
        
        # Check loss dictionary
        assert 'loss' in losses
        assert 'fidelity_loss' in losses
        assert 'kl_loss' in losses
        assert 'fidelity' in losses
        
        # Check all losses are finite
        for key, value in losses.items():
            assert np.isfinite(value), f"{key} is not finite"
    
    def test_train_with_early_stopping(self, config, model_and_data):
        """Test full training loop with early stopping"""
        model, data = model_and_data
        trainer = QVAETrainer(model, config)
        
        # Create train/val split
        from torch.utils.data import DataLoader, TensorDataset, random_split
        dataset = TensorDataset(data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Train
        history = trainer.train(train_loader, val_loader)
        
        # Check history
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_fidelity' in history
        assert 'val_fidelity' in history
        assert 'learning_rate' in history
        
        # Check history has entries
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0
        
        # Check all values are finite
        for key in ['train_loss', 'val_loss', 'train_fidelity', 'val_fidelity']:
            assert all(np.isfinite(v) for v in history[key])


class TestQVAEModule:
    """Test QVAEModule class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = Config.from_yaml("configs/default_config.yaml")
        # Reduce epochs for faster testing
        config.training.max_epochs = 5
        config.training.patience = 3
        return config
    
    @pytest.fixture
    def ground_states(self):
        """Create test ground states"""
        L = 4
        j2_j1_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        states = {}
        for j2_j1 in j2_j1_values:
            ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
            ham.build_hamiltonian()
            state = ham.compute_ground_state()
            states[(j2_j1, L)] = state
        
        return states
    
    def test_module_initialization(self, config):
        """Test module initializes correctly"""
        module = QVAEModule(config)
        
        assert module.config is config
        assert len(module.models) == 0
        assert len(module.trainers) == 0
        assert len(module.training_histories) == 0
    
    def test_prepare_dataset(self, config, ground_states):
        """Test dataset preparation"""
        module = QVAEModule(config)
        
        L = 4
        train_loader, val_loader = module.prepare_dataset(ground_states, L)
        
        # Check loaders are created
        assert train_loader is not None
        assert val_loader is not None
        
        # Check data can be loaded
        train_batch = next(iter(train_loader))
        assert len(train_batch) > 0
    
    def test_train_for_lattice_size(self, config, ground_states):
        """Test training for single lattice size"""
        module = QVAEModule(config)
        
        L = 4
        model = module.train_for_lattice_size(ground_states, L)
        
        # Check model is created and stored
        assert model is not None
        assert L in module.models
        assert L in module.trainers
        assert L in module.training_histories
        
        # Check training history
        history = module.training_histories[L]
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0
    
    def test_encode_all(self, config, ground_states):
        """Test encoding all states"""
        module = QVAEModule(config)
        
        # Train model first
        L = 4
        module.train_for_lattice_size(ground_states, L)
        
        # Encode all states
        latent_reps = module.encode_all(ground_states)
        
        # Check all states are encoded
        assert len(latent_reps) == len(ground_states)
        
        # Check latent representations have correct shape
        for (j2_j1, l), z in latent_reps.items():
            assert z.shape == (config.qvae_architecture.latent_dim,)
            assert np.all(np.isfinite(z))
    
    def test_save_and_load_models(self, config, ground_states):
        """Test saving and loading models"""
        module = QVAEModule(config)
        
        # Train model
        L = 4
        module.train_for_lattice_size(ground_states, L)
        
        # Save models
        with tempfile.TemporaryDirectory() as tmpdir:
            module.save_models(tmpdir)
            
            # Check checkpoint file exists
            checkpoint_path = Path(tmpdir) / f"qvae_L{L}.pt"
            assert checkpoint_path.exists()
            
            # Create new module and load
            module2 = QVAEModule(config)
            module2.load_models(tmpdir)
            
            # Check model is loaded
            assert L in module2.models
            
            # Check models produce same output
            sample_state = ground_states[(0.5, L)]
            x = torch.from_numpy(sample_state.to_real_vector()).float().unsqueeze(0)
            
            # Get device from model
            device1 = next(module.models[L].parameters()).device
            device2 = next(module2.models[L].parameters()).device
            
            with torch.no_grad():
                z1 = module.models[L].encode(x.to(device1))
                z2 = module2.models[L].encode(x.to(device2))
            
            # Move to CPU for comparison
            assert torch.allclose(z1.cpu(), z2.cpu(), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
