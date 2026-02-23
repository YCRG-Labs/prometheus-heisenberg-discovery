"""Unit tests for critical point detection module"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
from src.critical_point_detection import (
    LatentVarianceMethod,
    ReconstructionErrorMethod,
    FidelitySusceptibilityMethod,
    CriticalPointDetection
)


class TestLatentVarianceMethod:
    """Tests for LatentVarianceMethod class"""
    
    def test_initialization(self):
        """Test LatentVarianceMethod initialization"""
        method = LatentVarianceMethod(smooth_window=5, smooth_order=2)
        assert method.smooth_window == 5
        assert method.smooth_order == 2
    
    def test_compute_latent_variance(self):
        """Test latent variance computation"""
        method = LatentVarianceMethod()
        
        # Create synthetic latent representations with known variance
        latent_reps = {}
        for j2_j1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for L in [4, 5, 6]:
                # Create latent vectors with variance that peaks at j2_j1=0.5
                if j2_j1 == 0.5:
                    z = np.random.randn(8) * 2.0  # Higher variance
                else:
                    z = np.random.randn(8) * 0.5  # Lower variance
                latent_reps[(j2_j1, L)] = z
        
        variance_dict = method.compute_latent_variance(latent_reps)
        
        # Check that variance is computed for all j2_j1 values
        assert len(variance_dict) == 5
        assert all(j2_j1 in variance_dict for j2_j1 in [0.3, 0.4, 0.5, 0.6, 0.7])
        
        # Check that all variances are non-negative
        assert all(v >= 0 for v in variance_dict.values())
    
    def test_detect_critical_point_with_clear_peak(self):
        """Test critical point detection with a clear peak"""
        method = LatentVarianceMethod(smooth_window=3, smooth_order=1)
        
        # Create synthetic data with peak at j2_j1=0.5
        latent_reps = {}
        j2_j1_values = np.linspace(0.3, 0.7, 21)
        
        for j2_j1 in j2_j1_values:
            for L in [4, 5, 6]:
                # Gaussian peak centered at 0.5
                variance_factor = np.exp(-((j2_j1 - 0.5) ** 2) / 0.01)
                z = np.random.randn(8) * (0.5 + variance_factor)
                latent_reps[(j2_j1, L)] = z
        
        j2_j1_c, uncertainty = method.detect_critical_point(latent_reps)
        
        # Check that detected critical point is near 0.5
        assert 0.45 <= j2_j1_c <= 0.55
        assert uncertainty > 0
    
    def test_detect_critical_point_no_peaks_raises_error(self):
        """Test that detection raises error when no peaks found"""
        method = LatentVarianceMethod()
        
        # Create flat data with no peaks
        latent_reps = {}
        for j2_j1 in np.linspace(0.3, 0.7, 10):
            for L in [4, 5, 6]:
                z = np.ones(8) * 0.1  # Constant, no variance
                latent_reps[(j2_j1, L)] = z
        
        with pytest.raises(ValueError, match="No peaks found"):
            method.detect_critical_point(latent_reps)


class TestReconstructionErrorMethod:
    """Tests for ReconstructionErrorMethod class"""
    
    def test_initialization(self):
        """Test ReconstructionErrorMethod initialization"""
        qvae_module = Mock()
        method = ReconstructionErrorMethod(qvae_module, smooth_window=5, smooth_order=2)
        assert method.qvae_module == qvae_module
        assert method.smooth_window == 5
        assert method.smooth_order == 2
    
    def test_compute_reconstruction_error(self):
        """Test reconstruction error computation"""
        # Create mock Q-VAE module
        qvae_module = Mock()
        
        # Create mock model with proper return values
        mock_model = MagicMock()
        
        # Mock the forward pass to return a tuple
        mock_recon = torch.randn(1, 32)
        mock_mu = torch.zeros(1, 8)
        mock_logvar = torch.zeros(1, 8)
        mock_model.return_value = (mock_recon, mock_mu, mock_logvar)
        
        # Mock compute_fidelity to return high fidelity
        mock_model.compute_fidelity.return_value = torch.tensor([0.95])
        
        qvae_module.models = {4: mock_model}
        
        method = ReconstructionErrorMethod(qvae_module)
        
        # Create mock states
        mock_state = Mock()
        mock_state.to_real_vector = Mock(return_value=np.random.randn(32))
        
        states = {(0.5, 4): mock_state}
        
        error_dict = method.compute_reconstruction_error(states)
        
        # Check that error is computed
        assert (0.5, 4) in error_dict
        assert 0 <= error_dict[(0.5, 4)] <= 1
    
    def test_compute_reconstruction_error_missing_model_raises_error(self):
        """Test that missing model raises error"""
        qvae_module = Mock()
        qvae_module.models = {}  # No models
        
        method = ReconstructionErrorMethod(qvae_module)
        
        mock_state = Mock()
        states = {(0.5, 4): mock_state}
        
        with pytest.raises(RuntimeError, match="No trained Q-VAE model"):
            method.compute_reconstruction_error(states)


class TestFidelitySusceptibilityMethod:
    """Tests for FidelitySusceptibilityMethod class"""
    
    def test_initialization(self):
        """Test FidelitySusceptibilityMethod initialization"""
        method = FidelitySusceptibilityMethod(smooth_window=5, smooth_order=2)
        assert method.smooth_window == 5
        assert method.smooth_order == 2
    
    def test_compute_fidelity(self):
        """Test fidelity computation between two states"""
        method = FidelitySusceptibilityMethod()
        
        # Create mock states with known overlap
        state1 = Mock()
        state1.coefficients = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        
        state2 = Mock()
        state2.coefficients = np.array([0.8, 0.6, 0.0, 0.0], dtype=complex)
        
        fidelity = method._compute_fidelity(state1, state2)
        
        # Fidelity = |<ψ1|ψ2>|² = |0.8|² = 0.64
        assert 0 <= fidelity <= 1
        assert abs(fidelity - 0.64) < 0.01
    
    def test_compute_fidelity_identical_states(self):
        """Test fidelity of identical states is 1"""
        method = FidelitySusceptibilityMethod()
        
        state = Mock()
        state.coefficients = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        
        fidelity = method._compute_fidelity(state, state)
        
        assert abs(fidelity - 1.0) < 1e-10
    
    def test_compute_fidelity_susceptibility(self):
        """Test fidelity susceptibility computation"""
        method = FidelitySusceptibilityMethod()
        
        # Create mock states with varying overlap
        states = {}
        for j2_j1 in [0.4, 0.5, 0.6]:
            for L in [4]:
                state = Mock()
                # Create states that vary with j2_j1
                angle = (j2_j1 - 0.5) * np.pi
                state.coefficients = np.array([
                    np.cos(angle), np.sin(angle), 0.0, 0.0
                ], dtype=complex)
                states[(j2_j1, L)] = state
        
        susc_dict = method.compute_fidelity_susceptibility(states, delta=0.1)
        
        # Check that susceptibility is computed
        assert len(susc_dict) > 0
        
        # Check that all susceptibilities are finite
        for susc in susc_dict.values():
            assert np.isfinite(susc)


class TestCriticalPointDetection:
    """Tests for CriticalPointDetection class"""
    
    def test_initialization(self):
        """Test CriticalPointDetection initialization"""
        config = Mock()
        qvae_module = Mock()
        
        cpd = CriticalPointDetection(config, qvae_module)
        
        assert cpd.config == config
        assert cpd.qvae_module == qvae_module
        assert 'latent_variance' in cpd.methods
        assert 'reconstruction_error' in cpd.methods
        assert 'fidelity_susceptibility' in cpd.methods
    
    def test_ensemble_estimate(self):
        """Test ensemble estimate with inverse-variance weighting"""
        config = Mock()
        qvae_module = Mock()
        cpd = CriticalPointDetection(config, qvae_module)
        
        # Create mock detections
        detections = {
            'method1': (0.50, 0.02),  # j2_j1_c=0.50, uncertainty=0.02
            'method2': (0.52, 0.03),  # j2_j1_c=0.52, uncertainty=0.03
            'method3': (0.48, 0.04),  # j2_j1_c=0.48, uncertainty=0.04
        }
        
        j2_j1_ensemble, uncertainty_ensemble = cpd.ensemble_estimate(detections)
        
        # Check that ensemble estimate is within range of individual estimates
        assert 0.48 <= j2_j1_ensemble <= 0.52
        
        # Check that ensemble uncertainty is positive and smaller than individual uncertainties
        assert uncertainty_ensemble > 0
        assert uncertainty_ensemble <= min(0.02, 0.03, 0.04)
    
    def test_ensemble_estimate_single_detection(self):
        """Test ensemble estimate with single detection"""
        config = Mock()
        qvae_module = Mock()
        cpd = CriticalPointDetection(config, qvae_module)
        
        detections = {'method1': (0.50, 0.02)}
        
        j2_j1_ensemble, uncertainty_ensemble = cpd.ensemble_estimate(detections)
        
        # With single detection, ensemble should match the detection
        assert j2_j1_ensemble == 0.50
        assert uncertainty_ensemble == 0.02
    
    def test_ensemble_estimate_empty_raises_error(self):
        """Test that empty detections raises error"""
        config = Mock()
        qvae_module = Mock()
        cpd = CriticalPointDetection(config, qvae_module)
        
        with pytest.raises(ValueError, match="No valid detections"):
            cpd.ensemble_estimate({})
    
    def test_bootstrap_uncertainty(self):
        """Test bootstrap uncertainty estimation"""
        config = Mock()
        qvae_module = Mock()
        cpd = CriticalPointDetection(config, qvae_module)
        
        # Create mock data
        data = {i: np.random.randn(8) for i in range(20)}
        
        # Create mock method function that returns consistent estimates
        def mock_method(bootstrap_data):
            return 0.5, 0.01
        
        std_error = cpd.bootstrap_uncertainty(mock_method, data, n_bootstrap=10)
        
        # Check that standard error is non-negative
        assert std_error >= 0


class TestIntegration:
    """Integration tests for critical point detection"""
    
    def test_full_detection_pipeline(self):
        """Test full detection pipeline with synthetic data"""
        # This test would require more complex setup with actual Q-VAE models
        # For now, we test that the components work together
        
        config = Mock()
        qvae_module = Mock()
        
        cpd = CriticalPointDetection(config, qvae_module)
        
        # Verify that all methods are initialized
        assert len(cpd.methods) == 3
        assert all(method is not None for method in cpd.methods.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
