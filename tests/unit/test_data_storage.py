"""Unit tests for data storage module"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from src.data_storage import DataStorage
from src.ed_module import GroundState
from src.config import Config


@pytest.fixture
def temp_config():
    """Create temporary configuration for testing"""
    temp_dir = tempfile.mkdtemp()
    
    config = Config()
    config.paths.data_dir = temp_dir
    config.paths.output_dir = str(Path(temp_dir) / "output")
    config.paths.checkpoint_dir = str(Path(temp_dir) / "checkpoints")
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def storage(temp_config):
    """Create DataStorage instance with temporary directory"""
    return DataStorage(temp_config)


@pytest.fixture
def sample_ground_state():
    """Create a sample ground state for testing"""
    # Create a simple normalized wavefunction
    dim = 10
    coefficients = np.random.randn(dim) + 1j * np.random.randn(dim)
    coefficients = coefficients / np.linalg.norm(coefficients)
    
    metadata = {
        'convergence_tol': 1e-10,
        'maxiter': 1000,
        'computation_time': 1.5,
        'hilbert_dim': dim,
        'converged': True
    }
    
    return GroundState(
        coefficients=coefficients,
        energy=-5.2,
        basis=None,
        j2_j1=0.5,
        L=4,
        metadata=metadata
    )


@pytest.fixture
def sample_observables():
    """Create sample observables DataFrame"""
    data = {
        'j2_j1': [0.3, 0.4, 0.5, 0.6],
        'L': [4, 4, 4, 4],
        'energy': [-5.1, -5.2, -5.3, -5.4],
        'staggered_mag': [0.4, 0.3, 0.2, 0.1],
        'stripe_order': [0.1, 0.2, 0.3, 0.4]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_latent_reps():
    """Create sample latent representations"""
    latent_dim = 8
    return {
        (0.3, 4): np.random.randn(latent_dim),
        (0.4, 4): np.random.randn(latent_dim),
        (0.5, 4): np.random.randn(latent_dim),
    }


class SimpleQVAE(nn.Module):
    """Simple Q-VAE model for testing"""
    def __init__(self, input_dim=20, latent_dim=8):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


class TestDataStorageInitialization:
    """Test DataStorage initialization"""
    
    def test_initialization(self, storage, temp_config):
        """Test that DataStorage initializes correctly"""
        assert storage.data_dir == Path(temp_config.paths.data_dir)
        assert storage.hdf5_file.exists()
    
    def test_hdf5_structure(self, storage):
        """Test that HDF5 file has correct structure"""
        import h5py
        
        with h5py.File(storage.hdf5_file, 'r') as f:
            assert 'ground_states' in f
            assert 'observables' in f
            assert 'latent_representations' in f
            assert 'metadata' in f
            
            # Check metadata attributes
            metadata_group = f['metadata']
            assert 'numpy_version' in metadata_group.attrs
            assert 'pandas_version' in metadata_group.attrs
            assert 'h5py_version' in metadata_group.attrs
            assert 'torch_version' in metadata_group.attrs


class TestGroundStateStorage:
    """Test ground state save/load operations"""
    
    def test_save_ground_state(self, storage, sample_ground_state):
        """Test saving a ground state"""
        storage.save_ground_state(sample_ground_state, 0.5, 4)
        
        # Verify file was created and contains data
        import h5py
        with h5py.File(storage.hdf5_file, 'r') as f:
            gs_group = f['ground_states']
            key = storage._get_ground_state_key(0.5, 4)
            assert key in gs_group
    
    def test_load_ground_state(self, storage, sample_ground_state):
        """Test loading a ground state"""
        # Save first
        storage.save_ground_state(sample_ground_state, 0.5, 4)
        
        # Load
        loaded_state = storage.load_ground_state(0.5, 4)
        
        # Verify data matches
        assert loaded_state.L == 4
        assert loaded_state.j2_j1 == 0.5
        assert np.allclose(loaded_state.energy, sample_ground_state.energy)
        assert np.allclose(loaded_state.coefficients, sample_ground_state.coefficients)
    
    def test_ground_state_round_trip(self, storage, sample_ground_state):
        """Test that save/load preserves wavefunction to machine precision"""
        # Save
        storage.save_ground_state(sample_ground_state, 0.5, 4)
        
        # Load
        loaded_state = storage.load_ground_state(0.5, 4)
        
        # Check coefficients match to machine precision
        diff = np.abs(loaded_state.coefficients - sample_ground_state.coefficients)
        assert np.all(diff < 1e-14), "Wavefunction not preserved to machine precision"
        
        # Check normalization preserved
        assert abs(loaded_state.norm() - 1.0) < 1e-14
    
    def test_load_nonexistent_ground_state(self, storage):
        """Test loading a ground state that doesn't exist"""
        with pytest.raises(KeyError):
            storage.load_ground_state(0.99, 4)
    
    def test_save_invalid_ground_state(self, storage):
        """Test that saving invalid ground state raises error"""
        # Create unnormalized state
        coefficients = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        
        invalid_state = GroundState(
            coefficients=coefficients,
            energy=-1.0,
            basis=None,
            j2_j1=0.5,
            L=4,
            metadata={}
        )
        
        with pytest.raises(ValueError):
            storage.save_ground_state(invalid_state, 0.5, 4)
    
    def test_multiple_ground_states(self, storage):
        """Test saving and loading multiple ground states"""
        states = []
        params = [(0.3, 4), (0.4, 4), (0.5, 5)]
        
        for j2_j1, L in params:
            dim = 10 if L == 4 else 15
            coefficients = np.random.randn(dim) + 1j * np.random.randn(dim)
            coefficients = coefficients / np.linalg.norm(coefficients)
            
            state = GroundState(
                coefficients=coefficients,
                energy=-5.0 - j2_j1,
                basis=None,
                j2_j1=j2_j1,
                L=L,
                metadata={}
            )
            states.append(state)
            storage.save_ground_state(state, j2_j1, L)
        
        # Load and verify all states
        for i, (j2_j1, L) in enumerate(params):
            loaded = storage.load_ground_state(j2_j1, L)
            assert np.allclose(loaded.coefficients, states[i].coefficients)


class TestObservableStorage:
    """Test observable save/load operations"""
    
    def test_save_observables(self, storage, sample_observables):
        """Test saving observables DataFrame"""
        storage.save_observables(sample_observables)
        
        # Verify data was saved
        import h5py
        with h5py.File(storage.hdf5_file, 'r') as f:
            assert 'data' in f['observables']
    
    def test_load_observables(self, storage, sample_observables):
        """Test loading observables DataFrame"""
        # Save first
        storage.save_observables(sample_observables)
        
        # Load
        loaded_obs = storage.load_observables()
        
        # Verify data matches
        assert len(loaded_obs) == len(sample_observables)
        assert list(loaded_obs.columns) == list(sample_observables.columns)
        
        # Check values match
        for col in sample_observables.columns:
            assert np.allclose(loaded_obs[col], sample_observables[col])
    
    def test_save_empty_observables(self, storage):
        """Test that saving empty DataFrame raises error"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            storage.save_observables(empty_df)
    
    def test_save_observables_missing_columns(self, storage):
        """Test that saving DataFrame without required columns raises error"""
        df = pd.DataFrame({'energy': [1.0, 2.0]})
        
        with pytest.raises(ValueError):
            storage.save_observables(df)
    
    def test_load_nonexistent_observables(self, storage):
        """Test loading observables when none exist"""
        with pytest.raises(KeyError):
            storage.load_observables()


class TestLatentRepresentationStorage:
    """Test latent representation save/load operations"""
    
    def test_save_latent_representations(self, storage, sample_latent_reps):
        """Test saving latent representations"""
        storage.save_latent_representations(sample_latent_reps)
        
        # Verify data was saved
        import h5py
        with h5py.File(storage.hdf5_file, 'r') as f:
            latent_group = f['latent_representations']
            assert len(latent_group.keys()) == len(sample_latent_reps)
    
    def test_load_latent_representations(self, storage, sample_latent_reps):
        """Test loading latent representations"""
        # Save first
        storage.save_latent_representations(sample_latent_reps)
        
        # Load
        loaded_reps = storage.load_latent_representations()
        
        # Verify data matches
        assert len(loaded_reps) == len(sample_latent_reps)
        
        for key in sample_latent_reps:
            assert key in loaded_reps
            assert np.allclose(loaded_reps[key], sample_latent_reps[key])
    
    def test_latent_representation_consistency(self, storage, sample_latent_reps):
        """Test that latent representations are stored and loaded consistently"""
        # Save
        storage.save_latent_representations(sample_latent_reps)
        
        # Load multiple times
        loaded1 = storage.load_latent_representations()
        loaded2 = storage.load_latent_representations()
        
        # Verify consistency
        for key in sample_latent_reps:
            assert np.array_equal(loaded1[key], loaded2[key])
    
    def test_save_empty_latent_representations(self, storage):
        """Test that saving empty dict raises error"""
        with pytest.raises(ValueError):
            storage.save_latent_representations({})
    
    def test_load_nonexistent_latent_representations(self, storage):
        """Test loading latent representations when none exist"""
        with pytest.raises(KeyError):
            storage.load_latent_representations()


class TestQVAEModelStorage:
    """Test Q-VAE model checkpoint save/load operations"""
    
    def test_save_qvae_model(self, storage):
        """Test saving Q-VAE model checkpoint"""
        model = SimpleQVAE()
        metadata = {
            'latent_dim': 8,
            'learning_rate': 0.001,
            'epochs': 100
        }
        
        storage.save_qvae_model(model, L=4, metadata=metadata)
        
        # Verify checkpoint file exists
        checkpoint_path = storage.data_dir / "qvae_checkpoints" / "qvae_L4.pt"
        assert checkpoint_path.exists()
    
    def test_load_qvae_model(self, storage):
        """Test loading Q-VAE model checkpoint"""
        # Create and save model
        model = SimpleQVAE()
        original_state = model.state_dict()
        
        metadata = {'latent_dim': 8}
        storage.save_qvae_model(model, L=4, metadata=metadata)
        
        # Create new model and load checkpoint
        new_model = SimpleQVAE()
        loaded_metadata = storage.load_qvae_model(new_model, L=4)
        
        # Verify state dict matches
        new_state = new_model.state_dict()
        for key in original_state:
            assert torch.allclose(new_state[key], original_state[key])
        
        # Verify metadata
        assert loaded_metadata['latent_dim'] == 8
        assert loaded_metadata['L'] == 4
    
    def test_model_checkpoint_restoration(self, storage):
        """Test that model checkpoint restores identical parameters"""
        # Create model with random weights
        model = SimpleQVAE()
        
        # Get original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Save
        storage.save_qvae_model(model, L=4)
        
        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param))
        
        # Load checkpoint
        storage.load_qvae_model(model, L=4)
        
        # Verify parameters restored
        for name, param in model.named_parameters():
            assert torch.allclose(param, original_params[name])
    
    def test_load_nonexistent_model(self, storage):
        """Test loading model when checkpoint doesn't exist"""
        model = SimpleQVAE()
        
        with pytest.raises(FileNotFoundError):
            storage.load_qvae_model(model, L=99)


class TestMetadataStorage:
    """Test metadata save/load operations"""
    
    def test_save_metadata(self, storage):
        """Test saving arbitrary metadata"""
        metadata = {
            'experiment_name': 'test_run',
            'random_seed': 42,
            'parameters': [0.3, 0.4, 0.5]
        }
        
        storage.save_metadata('experiment_info', metadata)
        
        # Verify saved
        import h5py
        with h5py.File(storage.hdf5_file, 'r') as f:
            assert 'experiment_info' in f['metadata']
    
    def test_load_metadata(self, storage):
        """Test loading metadata"""
        metadata = {
            'experiment_name': 'test_run',
            'random_seed': 42
        }
        
        storage.save_metadata('experiment_info', metadata)
        loaded = storage.load_metadata('experiment_info')
        
        assert loaded == metadata
    
    def test_load_nonexistent_metadata(self, storage):
        """Test loading metadata that doesn't exist"""
        with pytest.raises(KeyError):
            storage.load_metadata('nonexistent_key')


class TestStorageInfo:
    """Test storage information retrieval"""
    
    def test_get_storage_info_empty(self, storage):
        """Test getting storage info for empty storage"""
        info = storage.get_storage_info()
        
        assert 'hdf5_file' in info
        assert 'file_size_mb' in info
        assert info['n_ground_states'] == 0
        assert info['n_observables'] == 0
        assert info['n_latent_representations'] == 0
    
    def test_get_storage_info_with_data(
        self,
        storage,
        sample_ground_state,
        sample_observables,
        sample_latent_reps
    ):
        """Test getting storage info with data"""
        # Add some data
        storage.save_ground_state(sample_ground_state, 0.5, 4)
        storage.save_observables(sample_observables)
        storage.save_latent_representations(sample_latent_reps)
        
        model = SimpleQVAE()
        storage.save_qvae_model(model, L=4)
        
        info = storage.get_storage_info()
        
        assert info['n_ground_states'] == 1
        assert info['n_observables'] == len(sample_observables)
        assert info['n_latent_representations'] == len(sample_latent_reps)
        assert info['n_qvae_checkpoints'] == 1
        assert info['file_size_mb'] > 0


class TestDataValidation:
    """Test data validation checks"""
    
    def test_ground_state_key_format(self, storage):
        """Test ground state key formatting"""
        key = storage._get_ground_state_key(0.5, 4)
        assert key == "L4_j2j1_0.500000"
        
        key = storage._get_ground_state_key(0.123456789, 5)
        assert key == "L5_j2j1_0.123457"
    
    def test_observables_required_columns(self, storage):
        """Test that observables must have required columns"""
        # Missing L column
        df = pd.DataFrame({'j2_j1': [0.5], 'energy': [-5.0]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            storage.save_observables(df)
