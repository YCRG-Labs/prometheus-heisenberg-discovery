"""Unit tests for Exact Diagonalization Module"""

import pytest
import numpy as np
from src.ed_module import J1J2Hamiltonian, GroundState, EDModule
from src.config import Config
from src.exceptions import ValidationError, NormalizationError, HermitianError


class TestJ1J2Hamiltonian:
    """Test J1J2Hamiltonian class"""
    
    def test_initialization(self):
        """Test Hamiltonian initialization"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        assert ham.L == 4
        assert ham.N == 16
        assert ham.j2_j1 == 0.5
        assert ham.basis is None
        assert ham.hamiltonian is None
    
    def test_invalid_lattice_size(self):
        """Test that invalid lattice sizes raise ValidationError"""
        with pytest.raises(ValidationError, match="not supported"):
            J1J2Hamiltonian(L=3, j2_j1=0.5)
    
    def test_negative_j2_j1(self):
        """Test that negative j2_j1 raises ValidationError"""
        with pytest.raises(ValidationError, match="must be non-negative"):
            J1J2Hamiltonian(L=4, j2_j1=-0.1)
    
    def test_build_hamiltonian(self):
        """Test Hamiltonian construction"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        H = ham.build_hamiltonian()
        
        # Check that Hamiltonian is constructed
        assert H is not None
        assert ham.hamiltonian is not None
        assert ham.basis is not None
        
        # Check that it's a sparse matrix
        assert hasattr(H, 'tocsr')
        
        # Check dimension is reasonable (should be reduced by symmetries)
        assert H.shape[0] == H.shape[1]
        assert H.shape[0] > 0
        assert H.shape[0] < 2**16  # Much smaller than full Hilbert space
    
    def test_hermiticity(self):
        """Test that constructed Hamiltonian is Hermitian"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        assert ham.verify_hermiticity(tol=1e-8)
    
    def test_j2_zero_case(self):
        """Test limiting case J2=0 (pure Néel model)"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.0)
        H = ham.build_hamiltonian()
        
        assert H is not None
        assert ham.verify_hermiticity()
    
    def test_compute_ground_state(self):
        """Test ground state computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        ground_state = ham.compute_ground_state(tol=1e-10)
        
        # Check ground state properties
        assert isinstance(ground_state, GroundState)
        assert ground_state.L == 4
        assert ground_state.j2_j1 == 0.5
        assert isinstance(ground_state.energy, float)
        assert len(ground_state.coefficients) == ham.basis.Ns
        
        # Check normalization
        assert abs(ground_state.norm() - 1.0) < 1e-8
        
        # Check validation passes
        assert ground_state.validate()


class TestGroundState:
    """Test GroundState class"""
    
    def test_initialization(self):
        """Test GroundState initialization"""
        # Create a simple normalized state
        coeffs = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        
        # Create a mock basis object
        class MockBasis:
            Ns = 2
        
        state = GroundState(
            coefficients=coeffs,
            energy=-1.0,
            basis=MockBasis(),
            j2_j1=0.5,
            L=4,
            metadata={'test': True}
        )
        
        assert state.L == 4
        assert state.j2_j1 == 0.5
        assert state.energy == -1.0
        assert state.metadata['test'] is True
    
    def test_to_real_vector(self):
        """Test conversion to real vector"""
        coeffs = np.array([1.0 + 2.0j, 3.0 + 4.0j])
        
        class MockBasis:
            Ns = 2
        
        state = GroundState(
            coefficients=coeffs,
            energy=-1.0,
            basis=MockBasis(),
            j2_j1=0.5,
            L=4
        )
        
        real_vec = state.to_real_vector()
        
        # Check length is 2 * original length
        assert len(real_vec) == 4
        
        # Check real and imaginary parts are correctly separated
        assert real_vec[0] == 1.0
        assert real_vec[1] == 3.0
        assert real_vec[2] == 2.0
        assert real_vec[3] == 4.0
    
    def test_norm(self):
        """Test norm computation"""
        # Create normalized state
        coeffs = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)])
        
        class MockBasis:
            Ns = 2
        
        state = GroundState(
            coefficients=coeffs,
            energy=-1.0,
            basis=MockBasis(),
            j2_j1=0.5,
            L=4
        )
        
        assert abs(state.norm() - 1.0) < 1e-10
    
    def test_validate(self):
        """Test validation method"""
        # Create normalized state
        coeffs = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)])
        
        class MockBasis:
            Ns = 2
        
        state = GroundState(
            coefficients=coeffs,
            energy=-1.0,
            basis=MockBasis(),
            j2_j1=0.5,
            L=4
        )
        
        assert state.validate()
        
        # Test with unnormalized state
        coeffs_bad = np.array([1.0, 1.0])
        state_bad = GroundState(
            coefficients=coeffs_bad,
            energy=-1.0,
            basis=MockBasis(),
            j2_j1=0.5,
            L=4
        )
        
        assert not state_bad.validate()


class TestEDModule:
    """Test EDModule class"""
    
    def test_initialization(self):
        """Test EDModule initialization"""
        config = Config.from_yaml("configs/default_config.yaml")
        ed_module = EDModule(config)
        
        assert ed_module.config is config
        assert ed_module.checkpoint_dir.exists()
    
    def test_compute_single_point(self):
        """Test single point computation"""
        config = Config.from_yaml("configs/default_config.yaml")
        ed_module = EDModule(config)
        
        key, state = ed_module._compute_single_point(j2_j1=0.5, L=4, tol=1e-10)
        
        assert key == (0.5, 4)
        assert isinstance(state, GroundState)
        assert state.L == 4
        assert state.j2_j1 == 0.5
        assert state.validate()
    
    def test_run_parameter_sweep_small(self):
        """Test parameter sweep on small subset"""
        # Create config with small parameter range
        config = Config.from_yaml("configs/default_config.yaml")
        config.ed_parameters.lattice_sizes = [4]
        config.ed_parameters.j2_j1_min = 0.4
        config.ed_parameters.j2_j1_max = 0.5
        config.ed_parameters.j2_j1_step = 0.1
        
        ed_module = EDModule(config)
        
        # Run sweep without parallelization for testing
        results = ed_module.run_parameter_sweep(parallel=False, resume=False)
        
        # Check that we got results for all points
        expected_points = config.get_parameter_points()
        assert len(results) == len(expected_points)
        
        # Check all results are valid
        for key, state in results.items():
            assert isinstance(state, GroundState)
            assert state.validate()
            assert key == (state.j2_j1, state.L)
