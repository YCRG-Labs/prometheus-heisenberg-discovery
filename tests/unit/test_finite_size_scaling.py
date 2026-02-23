"""
Unit tests for finite-size scaling module.
"""

import pytest
import numpy as np
from src.finite_size_scaling import FiniteSizeScaling


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.bootstrap_samples = 100  # Reduced for faster testing


@pytest.fixture
def fss():
    """Create FiniteSizeScaling instance."""
    config = MockConfig()
    return FiniteSizeScaling(config)


@pytest.fixture
def synthetic_scaling_data():
    """
    Generate synthetic data obeying known scaling form.
    
    Uses: O(λ, L) = L^(-x_O/ν) * f((λ - λ_c) * L^(1/ν))
    with f(x) = 1 / (1 + x^2) as the scaling function
    """
    # Known parameters
    j2_j1_c_true = 0.5
    nu_true = 1.0
    x_O_true = 0.25
    
    # Generate data for multiple system sizes
    L_values = [4, 5, 6]
    j2_j1_values = np.linspace(0.3, 0.7, 20)
    
    j2_j1_list = []
    L_list = []
    obs_list = []
    
    for L in L_values:
        for j2_j1 in j2_j1_values:
            # Compute scaled coordinate
            x_scaled = (j2_j1 - j2_j1_c_true) * L**(1.0 / nu_true)
            
            # Scaling function
            f_x = 1.0 / (1.0 + x_scaled**2)
            
            # Observable value
            obs = L**(-x_O_true / nu_true) * f_x
            
            # Add small noise
            obs += np.random.normal(0, 0.01 * obs)
            
            j2_j1_list.append(j2_j1)
            L_list.append(L)
            obs_list.append(obs)
    
    return {
        'j2_j1': np.array(j2_j1_list),
        'L': np.array(L_list),
        'observable': np.array(obs_list),
        'j2_j1_c_true': j2_j1_c_true,
        'nu_true': nu_true,
        'x_O_true': x_O_true
    }


def test_scaling_ansatz(fss):
    """Test scaling ansatz coordinate transformation."""
    j2_j1 = np.array([0.4, 0.5, 0.6])
    L = np.array([4, 5, 6])
    j2_j1_c = 0.5
    nu = 1.0
    
    x_scaled = fss.scaling_ansatz(j2_j1, L, j2_j1_c, nu, x_O=0.0)
    
    # Check expected values
    expected = np.array([-0.4, 0.0, 0.6])
    np.testing.assert_allclose(x_scaled, expected, rtol=1e-10)


def test_collapse_quality_perfect_collapse(fss):
    """Test collapse quality metric with perfect collapse."""
    # Create data that lies exactly on a curve
    x_scaled = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    y_scaled = np.array([1.0, 0.8, 0.5, 1.0, 0.8, 0.5])  # Perfect collapse
    L_values = np.array([4, 4, 4, 5, 5, 5])
    
    chi_sq = fss.collapse_quality(x_scaled, y_scaled, L_values)
    
    # Should be very small (near zero) for perfect collapse
    assert chi_sq < 0.1


def test_collapse_quality_poor_collapse(fss):
    """Test collapse quality metric with poor collapse."""
    # Create data with large scatter
    x_scaled = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    y_scaled = np.array([1.0, 0.8, 0.5, 2.0, 1.5, 1.0])  # Poor collapse
    L_values = np.array([4, 4, 4, 5, 5, 5])
    
    chi_sq = fss.collapse_quality(x_scaled, y_scaled, L_values)
    
    # Should be large for poor collapse
    assert chi_sq > 0.1


def test_optimize_collapse_synthetic_data(fss, synthetic_scaling_data):
    """Test optimization on synthetic data with known scaling."""
    data = synthetic_scaling_data
    
    result = fss.optimize_collapse(
        data['j2_j1'],
        data['L'],
        data['observable'],
        j2_j1_c_init=0.48,  # Slightly off initial guess
        method='differential_evolution'
    )
    
    # Check that optimization succeeded
    assert result['success']
    
    # Check that recovered parameters are in reasonable range
    # Note: With noise and finite data, exact recovery is not expected
    assert 0.3 <= result['j2_j1_c'] <= 0.7
    assert 0.3 <= result['nu'] <= 2.0
    assert -2.0 <= result['x_O'] <= 2.0
    
    # Check that chi_squared is finite and reasonable
    assert np.isfinite(result['chi_squared'])
    assert result['chi_squared'] < 1.0  # Should be relatively small for good fit


def test_optimize_collapse_invalid_inputs(fss):
    """Test that optimize_collapse handles invalid inputs."""
    # Mismatched array lengths
    with pytest.raises(ValueError, match="same length"):
        fss.optimize_collapse(
            np.array([0.4, 0.5]),
            np.array([4]),
            np.array([1.0, 2.0]),
            j2_j1_c_init=0.5
        )
    
    # Too few data points
    with pytest.raises(ValueError, match="at least 3"):
        fss.optimize_collapse(
            np.array([0.4, 0.5]),
            np.array([4, 5]),
            np.array([1.0, 2.0]),
            j2_j1_c_init=0.5
        )


def test_bootstrap_exponents(fss, synthetic_scaling_data):
    """Test bootstrap uncertainty estimation."""
    data = synthetic_scaling_data
    
    # Use small subset for faster test
    n_points = 30
    indices = np.random.choice(len(data['j2_j1']), n_points, replace=False)
    
    result = fss.bootstrap_exponents(
        data['j2_j1'][indices],
        data['L'][indices],
        data['observable'][indices],
        j2_j1_c_init=0.5,
        n_bootstrap=20  # Small number for fast test
    )
    
    # Check that all parameters have mean and std
    assert 'j2_j1_c' in result
    assert 'nu' in result
    assert 'x_O' in result
    
    # Check that uncertainties are positive
    assert result['j2_j1_c'][1] > 0
    assert result['nu'][1] > 0
    assert result['x_O'][1] > 0
    
    # Check that some bootstrap samples succeeded
    assert result['n_successful'] > 0


def test_extrapolate_to_thermodynamic_limit(fss):
    """Test extrapolation to L → ∞."""
    # Create synthetic finite-size critical points
    # j2_j1_c(L) = 0.5 + 0.1 * L^(-1)
    j2_j1_c_inf_true = 0.5
    nu = 1.0
    
    j2_j1_c_vs_L = {
        4: (0.5 + 0.1 * 4**(-1), 0.01),
        5: (0.5 + 0.1 * 5**(-1), 0.01),
        6: (0.5 + 0.1 * 6**(-1), 0.01)
    }
    
    j2_j1_c_inf, uncertainty = fss.extrapolate_to_thermodynamic_limit(
        j2_j1_c_vs_L, nu
    )
    
    # Check that extrapolated value is close to true value
    assert abs(j2_j1_c_inf - j2_j1_c_inf_true) < 0.02
    
    # Check that uncertainty is positive
    assert uncertainty > 0


def test_extrapolate_insufficient_data(fss):
    """Test that extrapolation requires at least 2 system sizes."""
    j2_j1_c_vs_L = {4: (0.5, 0.01)}
    
    with pytest.raises(ValueError, match="at least 2"):
        fss.extrapolate_to_thermodynamic_limit(j2_j1_c_vs_L, nu=1.0)


def test_get_scaled_data(fss):
    """Test transformation to scaled coordinates."""
    j2_j1 = np.array([0.4, 0.5, 0.6])
    L = np.array([4, 5, 6])
    observable = np.array([1.0, 0.8, 0.6])
    
    j2_j1_c = 0.5
    nu = 1.0
    x_O = 0.5
    
    x_scaled, y_scaled = fss.get_scaled_data(
        j2_j1, L, observable, j2_j1_c, nu, x_O
    )
    
    # Check x_scaled
    expected_x = np.array([-0.4, 0.0, 0.6])
    np.testing.assert_allclose(x_scaled, expected_x, rtol=1e-10)
    
    # Check y_scaled
    expected_y = observable * np.power(L, x_O / nu)
    np.testing.assert_allclose(y_scaled, expected_y, rtol=1e-10)


def test_collapse_quality_edge_cases(fss):
    """Test collapse quality with edge cases."""
    # Single point
    x = np.array([0.0])
    y = np.array([1.0])
    L = np.array([4])
    
    chi_sq = fss.collapse_quality(x, y, L)
    assert chi_sq == np.inf
    
    # All same x value
    x = np.array([0.5, 0.5, 0.5])
    y = np.array([1.0, 1.1, 0.9])
    L = np.array([4, 5, 6])
    
    chi_sq = fss.collapse_quality(x, y, L)
    assert chi_sq == np.inf
