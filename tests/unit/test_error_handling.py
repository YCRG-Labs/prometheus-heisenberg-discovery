"""
Unit tests for error handling and validation throughout the codebase.

Tests validation checks for:
- Wavefunction normalization
- Hamiltonian Hermiticity
- Parameter range validation
- NaN/Inf detection in computations
"""

import pytest
import numpy as np
import torch
from src.ed_module import J1J2Hamiltonian, GroundState
from src.qvae_module import QVAE
from src.config import Config
from src.exceptions import (
    ValidationError,
    NormalizationError,
    HermitianError,
    ConvergenceError,
    ComputationError
)


class TestParameterValidation:
    """Tests for parameter range validation."""
    
    def test_invalid_lattice_size(self):
        """Test that invalid lattice sizes are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            J1J2Hamiltonian(L=3, j2_j1=0.5)
        assert "not supported" in str(exc_info.value)
        assert exc_info.value.context['parameter'] == 'L'
    
    def test_negative_j2_j1(self):
        """Test that negative j2_j1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            J1J2Hamiltonian(L=4, j2_j1=-0.1)
        assert "non-negative" in str(exc_info.value)
        assert exc_info.value.context['parameter'] == 'j2_j1'
    
    def test_inf_j2_j1(self):
        """Test that infinite j2_j1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            J1J2Hamiltonian(L=4, j2_j1=np.inf)
        assert "finite" in str(exc_info.value)
        assert exc_info.value.context['parameter'] == 'j2_j1'
    
    def test_nan_j2_j1(self):
        """Test that NaN j2_j1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            J1J2Hamiltonian(L=4, j2_j1=np.nan)
        assert "finite" in str(exc_info.value)


class TestWavefunctionNormalization:
    """Tests for wavefunction normalization validation."""
    
    def test_unnormalized_wavefunction_rejected(self):
        """Test that unnormalized wavefunctions fail validation."""
        # Create a simple basis for testing
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        # Create unnormalized but finite coefficients
        dim = ham.basis.Ns
        coeffs = np.ones(dim, dtype=np.complex128) * 2.0  # Norm will be 2*sqrt(dim)
        # Normalize to avoid NaN/Inf check, but then corrupt
        coeffs = coeffs / np.linalg.norm(coeffs)
        
        gs = GroundState(
            coefficients=coeffs,
            energy=0.0,
            basis=ham.basis,
            j2_j1=0.5,
            L=4
        )
        
        # Manually corrupt normalization
        gs.coefficients = gs.coefficients * 2.0
        
        # Should fail validation
        assert not gs.validate(tol=1e-8)
    
    def test_nan_in_wavefunction_rejected(self):
        """Test that wavefunctions with NaN are rejected."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        dim = ham.basis.Ns
        coeffs = np.ones(dim, dtype=np.complex128)
        coeffs[0] = np.nan
        
        with pytest.raises(ValidationError) as exc_info:
            GroundState(
                coefficients=coeffs,
                energy=0.0,
                basis=ham.basis,
                j2_j1=0.5,
                L=4
            )
        assert "NaN or Inf" in str(exc_info.value)
    
    def test_inf_in_wavefunction_rejected(self):
        """Test that wavefunctions with Inf are rejected."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        dim = ham.basis.Ns
        coeffs = np.ones(dim, dtype=np.complex128)
        coeffs[0] = np.inf
        
        with pytest.raises(ValidationError) as exc_info:
            GroundState(
                coefficients=coeffs,
                energy=0.0,
                basis=ham.basis,
                j2_j1=0.5,
                L=4
            )
        assert "NaN or Inf" in str(exc_info.value)
    
    def test_nan_energy_rejected(self):
        """Test that NaN energy is rejected."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        dim = ham.basis.Ns
        coeffs = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        with pytest.raises(ValidationError) as exc_info:
            GroundState(
                coefficients=coeffs,
                energy=np.nan,
                basis=ham.basis,
                j2_j1=0.5,
                L=4
            )
        assert "NaN or Inf" in str(exc_info.value)


class TestHamiltonianHermiticity:
    """Tests for Hamiltonian Hermiticity validation."""
    
    def test_hermiticity_check(self):
        """Test that Hermiticity is verified during construction."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        H = ham.build_hamiltonian()
        
        # Verify Hermiticity
        assert ham.verify_hermiticity(tol=1e-8)
    
    def test_hermiticity_check_without_construction(self):
        """Test that Hermiticity check fails if Hamiltonian not constructed."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        
        with pytest.raises(ComputationError) as exc_info:
            ham.verify_hermiticity()
        assert "not constructed" in str(exc_info.value)


class TestNaNInfDetection:
    """Tests for NaN/Inf detection in computations."""
    
    def test_qvae_loss_nan_detection_input(self):
        """Test that NaN in input is detected in Q-VAE loss."""
        config = Config()
        hilbert_dim = 10
        qvae = QVAE(config, hilbert_dim)
        
        # Create inputs with NaN
        batch_size = 4
        x = torch.randn(batch_size, 2 * hilbert_dim)
        x[0, 0] = float('nan')
        
        recon = torch.randn(batch_size, 2 * hilbert_dim)
        mu = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        logvar = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        
        with pytest.raises(ComputationError) as exc_info:
            qvae.loss_function(x, recon, mu, logvar)
        assert "NaN or Inf" in str(exc_info.value)
    
    def test_qvae_loss_nan_detection_recon(self):
        """Test that NaN in reconstruction is detected in Q-VAE loss."""
        config = Config()
        hilbert_dim = 10
        qvae = QVAE(config, hilbert_dim)
        
        batch_size = 4
        x = torch.randn(batch_size, 2 * hilbert_dim)
        recon = torch.randn(batch_size, 2 * hilbert_dim)
        recon[0, 0] = float('nan')
        
        mu = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        logvar = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        
        with pytest.raises(ComputationError) as exc_info:
            qvae.loss_function(x, recon, mu, logvar)
        assert "NaN or Inf" in str(exc_info.value)
    
    def test_qvae_loss_nan_detection_mu(self):
        """Test that NaN in mu is detected in Q-VAE loss."""
        config = Config()
        hilbert_dim = 10
        qvae = QVAE(config, hilbert_dim)
        
        batch_size = 4
        x = torch.randn(batch_size, 2 * hilbert_dim)
        recon = torch.randn(batch_size, 2 * hilbert_dim)
        mu = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        mu[0, 0] = float('nan')
        logvar = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        
        with pytest.raises(ComputationError) as exc_info:
            qvae.loss_function(x, recon, mu, logvar)
        assert "NaN or Inf" in str(exc_info.value)
    
    def test_qvae_loss_inf_detection(self):
        """Test that Inf values are detected in Q-VAE loss."""
        config = Config()
        hilbert_dim = 10
        qvae = QVAE(config, hilbert_dim)
        
        batch_size = 4
        x = torch.randn(batch_size, 2 * hilbert_dim)
        recon = torch.randn(batch_size, 2 * hilbert_dim)
        mu = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        logvar = torch.randn(batch_size, config.qvae_architecture.latent_dim)
        logvar[0, 0] = float('inf')
        
        with pytest.raises(ComputationError) as exc_info:
            qvae.loss_function(x, recon, mu, logvar)
        assert "NaN or Inf" in str(exc_info.value)


class TestGroundStateValidation:
    """Tests for ground state validation."""
    
    def test_validate_normalized_state(self):
        """Test that properly normalized states pass validation."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        # Compute actual ground state
        gs = ham.compute_ground_state()
        
        # Should pass validation
        assert gs.validate(tol=1e-8)
    
    def test_validate_detects_unnormalized(self):
        """Test that validation detects unnormalized states."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        # Create normalized state first
        dim = ham.basis.Ns
        coeffs = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        gs = GroundState(
            coefficients=coeffs,
            energy=0.0,
            basis=ham.basis,
            j2_j1=0.5,
            L=4
        )
        
        # Manually corrupt normalization
        gs.coefficients = gs.coefficients * 2.0
        
        # Should fail validation
        assert not gs.validate(tol=1e-8)
    
    def test_validate_detects_nan(self):
        """Test that validation detects NaN in coefficients."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        dim = ham.basis.Ns
        coeffs = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        gs = GroundState(
            coefficients=coeffs,
            energy=0.0,
            basis=ham.basis,
            j2_j1=0.5,
            L=4
        )
        
        # Manually add NaN
        gs.coefficients[0] = np.nan
        
        # Should fail validation
        assert not gs.validate(tol=1e-8)
    
    def test_validate_detects_dimension_mismatch(self):
        """Test that validation detects dimension mismatch."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        
        dim = ham.basis.Ns
        coeffs = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        gs = GroundState(
            coefficients=coeffs,
            energy=0.0,
            basis=ham.basis,
            j2_j1=0.5,
            L=4
        )
        
        # Manually change coefficients to wrong dimension
        gs.coefficients = np.ones(dim + 1, dtype=np.complex128)
        
        # Should fail validation
        assert not gs.validate(tol=1e-8)


class TestExceptionContext:
    """Tests that exceptions include useful context information."""
    
    def test_validation_error_context(self):
        """Test that ValidationError includes context."""
        try:
            J1J2Hamiltonian(L=3, j2_j1=0.5)
        except ValidationError as e:
            assert e.context['parameter'] == 'L'
            assert e.context['expected'] == "{4, 5, 6}"
            assert e.context['actual'] == 3
    
    def test_computation_error_context(self):
        """Test that ComputationError includes context."""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        
        try:
            ham.verify_hermiticity()
        except ComputationError as e:
            assert 'operation' in e.context
            assert 'L' in e.context
            assert 'j2_j1' in e.context
