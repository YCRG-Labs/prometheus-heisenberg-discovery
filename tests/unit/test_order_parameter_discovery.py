"""
Unit tests for order parameter discovery module
"""

import pytest
import numpy as np
import pandas as pd
from src.order_parameter_discovery import (
    CorrelationAnalysis,
    OrderParameterDiscovery,
    CorrelationResult
)
from src.config import Config


@pytest.fixture
def config():
    """Create test configuration"""
    config = Config(
        lattice_sizes=[4, 5],
        j2_j1_min=0.3,
        j2_j1_max=0.7,
        j2_j1_step=0.1,
        latent_dim=4,
        correlation_threshold=0.8,
        bootstrap_samples=100,  # Reduced for faster tests
        significance_level=0.01
    )
    return config


@pytest.fixture
def correlation_analysis(config):
    """Create CorrelationAnalysis instance"""
    return CorrelationAnalysis(config)


@pytest.fixture
def order_parameter_discovery(config):
    """Create OrderParameterDiscovery instance"""
    return OrderParameterDiscovery(config)


class TestCorrelationAnalysis:
    """Tests for CorrelationAnalysis class"""
    
    def test_compute_pearson_correlation_perfect_positive(self, correlation_analysis):
        """Test perfect positive correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        
        r, p_value = correlation_analysis.compute_pearson_correlation(x, y)
        
        assert abs(r - 1.0) < 1e-10, f"Expected r=1.0, got {r}"
        assert p_value < 0.01, f"Expected significant p-value, got {p_value}"
    
    def test_compute_pearson_correlation_perfect_negative(self, correlation_analysis):
        """Test perfect negative correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        
        r, p_value = correlation_analysis.compute_pearson_correlation(x, y)
        
        assert abs(r - (-1.0)) < 1e-10, f"Expected r=-1.0, got {r}"
        assert p_value < 0.01, f"Expected significant p-value, got {p_value}"
    
    def test_compute_pearson_correlation_no_correlation(self, correlation_analysis):
        """Test zero correlation with random data"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        r, p_value = correlation_analysis.compute_pearson_correlation(x, y)
        
        # Should be close to zero (not exactly due to randomness)
        assert abs(r) < 0.3, f"Expected |r| < 0.3, got {r}"
        assert -1 <= r <= 1, f"Correlation {r} outside [-1, 1]"
        assert 0 <= p_value <= 1, f"P-value {p_value} outside [0, 1]"
    
    def test_compute_pearson_correlation_invalid_length(self, correlation_analysis):
        """Test error on mismatched array lengths"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="same length"):
            correlation_analysis.compute_pearson_correlation(x, y)
    
    def test_compute_pearson_correlation_insufficient_data(self, correlation_analysis):
        """Test error on insufficient data points"""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="at least 3 data points"):
            correlation_analysis.compute_pearson_correlation(x, y)
    
    def test_compute_pearson_correlation_nan_values(self, correlation_analysis):
        """Test error on NaN values"""
        x = np.array([1.0, 2.0, np.nan, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        with pytest.raises(ValueError, match="NaN or Inf"):
            correlation_analysis.compute_pearson_correlation(x, y)
    
    def test_compute_correlation_matrix_basic(self, correlation_analysis):
        """Test correlation matrix computation with synthetic data"""
        # Create synthetic latent data
        latent_data = pd.DataFrame({
            'j2_j1': [0.3, 0.4, 0.5, 0.6, 0.7],
            'L': [4, 4, 4, 4, 4],
            'z_0': [1.0, 2.0, 3.0, 4.0, 5.0],  # Perfect correlation with obs1
            'z_1': [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfect negative correlation with obs1
            'z_2': [1.0, 1.0, 1.0, 1.0, 1.0],  # Constant (no correlation)
        })
        
        # Create synthetic observable data (wide format)
        observable_data = pd.DataFrame({
            'j2_j1': [0.3, 0.4, 0.5, 0.6, 0.7],
            'L': [4, 4, 4, 4, 4],
            'obs1': [2.0, 4.0, 6.0, 8.0, 10.0],  # Correlated with z_0
            'obs2': [1.0, 1.0, 1.0, 1.0, 1.0],   # Constant
        })
        
        corr_matrix = correlation_analysis.compute_correlation_matrix(
            latent_data, observable_data
        )
        
        # Check shape
        assert corr_matrix.shape == (3, 2), f"Expected shape (3, 2), got {corr_matrix.shape}"
        
        # Check z_0 vs obs1 (should be ~1.0)
        assert abs(corr_matrix.loc['z_0', 'obs1'] - 1.0) < 0.01
        
        # Check z_1 vs obs1 (should be ~-1.0)
        assert abs(corr_matrix.loc['z_1', 'obs1'] - (-1.0)) < 0.01
        
        # Check z_2 vs obs1 (should be NaN due to constant z_2)
        # Note: correlation with constant is undefined
    
    def test_compute_correlation_matrix_long_format(self, correlation_analysis):
        """Test correlation matrix with long format observable data"""
        latent_data = pd.DataFrame({
            'j2_j1': [0.3, 0.4, 0.5],
            'L': [4, 4, 4],
            'z_0': [1.0, 2.0, 3.0],
        })
        
        # Long format observables
        observable_data = pd.DataFrame({
            'j2_j1': [0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
            'L': [4, 4, 4, 4, 4, 4],
            'observable_name': ['obs1', 'obs2', 'obs1', 'obs2', 'obs1', 'obs2'],
            'value': [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]
        })
        
        corr_matrix = correlation_analysis.compute_correlation_matrix(
            latent_data, observable_data
        )
        
        assert 'obs1' in corr_matrix.columns
        assert 'obs2' in corr_matrix.columns
        assert 'z_0' in corr_matrix.index
    
    def test_bootstrap_correlation_basic(self, correlation_analysis):
        """Test bootstrap correlation with confidence intervals"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        
        r, ci_lower, ci_upper = correlation_analysis.bootstrap_correlation(x, y, n_bootstrap=100)
        
        # Check correlation is close to 1.0
        assert abs(r - 1.0) < 0.01
        
        # Check CI ordering
        assert ci_lower <= r <= ci_upper, f"CI ordering violated: {ci_lower} <= {r} <= {ci_upper}"
        
        # Check CI bounds are reasonable
        assert ci_lower > 0.9, f"Lower CI {ci_lower} too low for perfect correlation"
        assert ci_upper <= 1.0, f"Upper CI {ci_upper} exceeds 1.0"
    
    def test_bootstrap_correlation_with_noise(self, correlation_analysis):
        """Test bootstrap correlation with noisy data"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.random.randn(50) * 2  # Linear with noise
        
        r, ci_lower, ci_upper = correlation_analysis.bootstrap_correlation(x, y, n_bootstrap=100)
        
        # Should have strong positive correlation
        assert r > 0.8
        
        # CI should be ordered
        assert ci_lower <= r <= ci_upper
        
        # CI width should be reasonable (not too wide)
        ci_width = ci_upper - ci_lower
        assert ci_width < 0.3, f"CI too wide: {ci_width}"
    
    def test_permutation_test_significant(self, correlation_analysis):
        """Test permutation test with significant correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        
        p_value = correlation_analysis.permutation_test(x, y, n_permutations=1000)
        
        # Should be highly significant
        assert p_value < 0.01, f"Expected p < 0.01, got {p_value}"
        assert 0 <= p_value <= 1, f"P-value {p_value} outside [0, 1]"
    
    def test_permutation_test_not_significant(self, correlation_analysis):
        """Test permutation test with no correlation"""
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)
        
        p_value = correlation_analysis.permutation_test(x, y, n_permutations=1000)
        
        # Should not be significant
        assert p_value > 0.05, f"Expected p > 0.05, got {p_value}"
        assert 0 <= p_value <= 1, f"P-value {p_value} outside [0, 1]"


class TestOrderParameterDiscovery:
    """Tests for OrderParameterDiscovery class"""
    
    def test_discover_order_parameters_synthetic(self, order_parameter_discovery):
        """Test order parameter discovery with synthetic data"""
        # Create synthetic latent representations
        latent_reps = {}
        for j2_j1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for L in [4, 5]:
                # z_0 correlates with j2_j1 (simulates order parameter)
                z = np.array([j2_j1, 1.0 - j2_j1, 0.5, 0.5])
                latent_reps[(j2_j1, L)] = z
        
        # Create synthetic observables
        obs_data = []
        for j2_j1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for L in [4, 5]:
                # staggered_mag decreases with j2_j1
                obs_data.append({
                    'j2_j1': j2_j1,
                    'L': L,
                    'observable_name': 'staggered_mag',
                    'value': 1.0 - j2_j1
                })
                # stripe_order increases with j2_j1
                obs_data.append({
                    'j2_j1': j2_j1,
                    'L': L,
                    'observable_name': 'stripe_order',
                    'value': j2_j1
                })
        
        observables = pd.DataFrame(obs_data)
        
        # Run discovery
        results = order_parameter_discovery.discover_order_parameters(
            latent_reps, observables
        )
        
        # Check results structure
        assert 'correlation_matrix' in results
        assert 'significant_correlations' in results
        assert 'discovered_order_parameters' in results
        assert 'validation_results' in results
        
        # Check correlation matrix
        corr_matrix = results['correlation_matrix']
        assert corr_matrix is not None
        assert 'z_0' in corr_matrix.index
        assert 'staggered_mag' in corr_matrix.columns or 'stripe_order' in corr_matrix.columns
    
    def test_validate_in_known_phases_neel(self, order_parameter_discovery):
        """Test validation in Néel phase"""
        # Create correlation matrix with strong staggered_mag correlation
        corr_matrix = pd.DataFrame({
            'staggered_mag': [0.95, -0.2, 0.1],
            'stripe_order': [0.1, 0.3, -0.85]
        }, index=['z_0', 'z_1', 'z_2'])
        
        # Create dummy observables
        observables = pd.DataFrame({
            'j2_j1': [0.3, 0.35],
            'L': [4, 4],
            'observable_name': ['staggered_mag', 'stripe_order'],
            'value': [0.8, 0.2]
        })
        
        results = order_parameter_discovery.validate_in_known_phases(
            corr_matrix, observables
        )
        
        # Should validate Néel phase
        assert results['neel_phase_valid'] == True
        assert results['neel_dominant_observable'] == 'staggered_mag'
    
    def test_validate_in_known_phases_stripe(self, order_parameter_discovery):
        """Test validation in stripe phase"""
        # Create correlation matrix with strong stripe_order correlation
        corr_matrix = pd.DataFrame({
            'staggered_mag': [0.1, -0.2, 0.1],
            'stripe_order': [0.1, 0.3, 0.92]
        }, index=['z_0', 'z_1', 'z_2'])
        
        # Create dummy observables
        observables = pd.DataFrame({
            'j2_j1': [0.65, 0.7],
            'L': [4, 4],
            'observable_name': ['staggered_mag', 'stripe_order'],
            'value': [0.2, 0.8]
        })
        
        results = order_parameter_discovery.validate_in_known_phases(
            corr_matrix, observables
        )
        
        # Should validate stripe phase
        assert results['stripe_phase_valid'] == True
        assert results['stripe_dominant_observable'] == 'stripe_order'
    
    def test_analyze_intermediate_regime(self, order_parameter_discovery):
        """Test intermediate regime analysis"""
        # Create latent representations with varying behavior in intermediate regime
        latent_reps = {}
        for j2_j1 in np.linspace(0.3, 0.7, 9):
            for L in [4, 5]:
                # z_0 has high variance in intermediate regime
                if 0.4 <= j2_j1 <= 0.6:
                    z_0 = np.sin(j2_j1 * 10)  # Oscillates
                else:
                    z_0 = 0.5  # Constant outside
                
                z = np.array([z_0, 0.5, 0.5, 0.5])
                latent_reps[(j2_j1, L)] = z
        
        # Create observables
        obs_data = []
        for j2_j1 in np.linspace(0.3, 0.7, 9):
            for L in [4, 5]:
                obs_data.append({
                    'j2_j1': j2_j1,
                    'L': L,
                    'observable_name': 'energy',
                    'value': -j2_j1
                })
        
        observables = pd.DataFrame(obs_data)
        
        # Analyze intermediate regime
        results = order_parameter_discovery.analyze_intermediate_regime(
            latent_reps, observables, j2_j1_range=(0.4, 0.6)
        )
        
        # Check results structure
        assert 'latent_variances' in results
        assert 'high_variance_dims' in results
        assert 'intermediate_correlations' in results
        assert 'phase_assessment' in results
        
        # z_0 should have higher variance than others
        variances = results['latent_variances']
        assert 'z_0' in variances
        assert variances['z_0'] > variances.get('z_1', 0)
    
    def test_analyze_intermediate_regime_empty(self, order_parameter_discovery):
        """Test intermediate regime analysis with no data"""
        latent_reps = {
            (0.3, 4): np.array([1.0, 2.0, 3.0, 4.0]),
            (0.7, 4): np.array([5.0, 6.0, 7.0, 8.0])
        }
        
        observables = pd.DataFrame({
            'j2_j1': [0.3, 0.7],
            'L': [4, 4],
            'observable_name': ['energy', 'energy'],
            'value': [-1.0, -2.0]
        })
        
        results = order_parameter_discovery.analyze_intermediate_regime(
            latent_reps, observables, j2_j1_range=(0.4, 0.6)
        )
        
        # Should handle empty data gracefully
        assert results['phase_assessment'] == 'insufficient_data'
        assert len(results['latent_variances']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
