"""
Unit tests for validation module

Tests validation functions for known phases including:
- Néel phase validation
- Stripe phase validation
- Phase separation validation
- Literature comparison
"""

import pytest
import numpy as np
import pandas as pd
from src.validation import (
    ValidationModule,
    PhaseValidationResult,
    PhaseSeparationResult,
    LiteratureComparisonResult
)


class TestValidationModule:
    """Test suite for ValidationModule"""
    
    @pytest.fixture
    def validation_module(self):
        """Create ValidationModule instance"""
        return ValidationModule()
    
    @pytest.fixture
    def sample_correlations(self):
        """Create sample correlation matrix"""
        # 3 latent dimensions × 4 observables
        data = {
            'staggered_mag': [0.85, 0.45, 0.20],
            'stripe_order': [0.30, 0.75, 0.40],
            'energy': [-0.60, -0.55, -0.50],
            'plaquette_order': [0.25, 0.35, 0.80]
        }
        index = ['z_0', 'z_1', 'z_2']
        return pd.DataFrame(data, index=index)
    
    @pytest.fixture
    def sample_observables_long(self):
        """Create sample observables in long format"""
        data = []
        # Néel regime (j2_j1 < 0.4)
        for j2_j1 in [0.30, 0.35]:
            for L in [4, 5]:
                data.append({'j2_j1': j2_j1, 'L': L, 'observable_name': 'staggered_mag', 'value': 0.4})
                data.append({'j2_j1': j2_j1, 'L': L, 'observable_name': 'stripe_order', 'value': 0.1})
                data.append({'j2_j1': j2_j1, 'L': L, 'observable_name': 'energy', 'value': -0.5})
        
        # Stripe regime (j2_j1 > 0.6)
        for j2_j1 in [0.65, 0.70]:
            for L in [4, 5]:
                data.append({'j2_j1': j2_j1, 'L': L, 'observable_name': 'staggered_mag', 'value': 0.1})
                data.append({'j2_j1': j2_j1, 'L': L, 'observable_name': 'stripe_order', 'value': 0.4})
                data.append({'j2_j1': j2_j1, 'L': L, 'observable_name': 'energy', 'value': -0.6})
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_latent_reps(self):
        """Create sample latent representations"""
        latent_reps = {}
        
        # Néel regime - cluster around origin
        for j2_j1 in [0.30, 0.35]:
            for L in [4, 5]:
                z = np.random.randn(3) * 0.5  # Small variance
                latent_reps[(j2_j1, L)] = z
        
        # Stripe regime - cluster away from origin
        for j2_j1 in [0.65, 0.70]:
            for L in [4, 5]:
                z = np.random.randn(3) * 0.5 + np.array([3.0, 3.0, 3.0])  # Shifted cluster
                latent_reps[(j2_j1, L)] = z
        
        return latent_reps
    
    def test_validate_neel_phase_success(self, validation_module, sample_correlations, sample_observables_long):
        """Test Néel phase validation with strong correlation"""
        result = validation_module.validate_neel_phase(
            sample_correlations,
            sample_observables_long
        )
        
        assert isinstance(result, PhaseValidationResult)
        assert result.phase_name == 'Neel'
        assert result.is_valid is True  # z_0 has 0.85 correlation with staggered_mag
        assert result.max_correlation >= 0.7
        assert result.dominant_observable == 'staggered_mag'
        assert result.dominant_latent_dim == 'z_0'
    
    def test_validate_neel_phase_failure(self, validation_module, sample_observables_long):
        """Test Néel phase validation with weak correlations"""
        # Create correlation matrix with weak correlations
        weak_correlations = pd.DataFrame({
            'staggered_mag': [0.3, 0.2, 0.1],
            'stripe_order': [0.2, 0.3, 0.1],
            'energy': [-0.4, -0.3, -0.2]
        }, index=['z_0', 'z_1', 'z_2'])
        
        result = validation_module.validate_neel_phase(
            weak_correlations,
            sample_observables_long
        )
        
        assert result.is_valid is False
        assert result.max_correlation < 0.7
    
    def test_validate_neel_phase_missing_observable(self, validation_module, sample_observables_long):
        """Test Néel phase validation when staggered_mag is missing"""
        # Correlation matrix without staggered_mag
        correlations = pd.DataFrame({
            'stripe_order': [0.3, 0.7, 0.4],
            'energy': [-0.6, -0.5, -0.4]
        }, index=['z_0', 'z_1', 'z_2'])
        
        result = validation_module.validate_neel_phase(
            correlations,
            sample_observables_long
        )
        
        assert result.is_valid is False
        assert 'error' in result.details
    
    def test_validate_stripe_phase_success(self, validation_module, sample_correlations, sample_observables_long):
        """Test stripe phase validation with strong correlation"""
        result = validation_module.validate_stripe_phase(
            sample_correlations,
            sample_observables_long
        )
        
        assert isinstance(result, PhaseValidationResult)
        assert result.phase_name == 'Stripe'
        assert result.is_valid is True  # z_1 has 0.75 correlation with stripe_order
        assert result.max_correlation >= 0.7
        assert result.dominant_observable == 'stripe_order'
        assert result.dominant_latent_dim == 'z_1'
    
    def test_validate_stripe_phase_failure(self, validation_module, sample_observables_long):
        """Test stripe phase validation with weak correlations"""
        weak_correlations = pd.DataFrame({
            'staggered_mag': [0.3, 0.2, 0.1],
            'stripe_order': [0.2, 0.3, 0.1],
            'energy': [-0.4, -0.3, -0.2]
        }, index=['z_0', 'z_1', 'z_2'])
        
        result = validation_module.validate_stripe_phase(
            weak_correlations,
            sample_observables_long
        )
        
        assert result.is_valid is False
        assert result.max_correlation < 0.7
    
    def test_validate_phase_separation_success(self, validation_module, sample_latent_reps, sample_observables_long):
        """Test phase separation with well-separated clusters"""
        result = validation_module.validate_phase_separation(
            sample_latent_reps,
            sample_observables_long
        )
        
        assert isinstance(result, PhaseSeparationResult)
        assert result.n_clusters == 2
        assert result.silhouette_score >= -1.0
        assert result.silhouette_score <= 1.0
        # With well-separated clusters, should have good silhouette score
        assert result.is_well_separated is True
    
    def test_validate_phase_separation_single_phase(self, validation_module, sample_observables_long):
        """Test phase separation with only one phase present"""
        # Only Néel regime data
        neel_only = {}
        for j2_j1 in [0.30, 0.35]:
            for L in [4, 5]:
                neel_only[(j2_j1, L)] = np.random.randn(3)
        
        result = validation_module.validate_phase_separation(
            neel_only,
            sample_observables_long
        )
        
        # Should fail validation due to missing stripe phase
        assert result.is_well_separated is False
    
    def test_validate_phase_separation_specific_L(self, validation_module, sample_latent_reps, sample_observables_long):
        """Test phase separation for specific lattice size"""
        result = validation_module.validate_phase_separation(
            sample_latent_reps,
            sample_observables_long,
            L=4
        )
        
        assert isinstance(result, PhaseSeparationResult)
        assert len(result.cluster_labels) > 0
    
    def test_compare_with_literature_consistent(self, validation_module):
        """Test literature comparison with consistent estimates"""
        estimated = {
            'neel_to_intermediate': 0.42,
            'intermediate_to_stripe': 0.58
        }
        
        results = validation_module.compare_with_literature(estimated)
        
        assert 'neel_to_intermediate' in results
        assert 'intermediate_to_stripe' in results
        
        # Both should be consistent with default literature ranges
        assert results['neel_to_intermediate'].is_consistent is True
        assert results['intermediate_to_stripe'].is_consistent is True
        assert results['neel_to_intermediate'].estimated_critical_point == 0.42
    
    def test_compare_with_literature_inconsistent(self, validation_module):
        """Test literature comparison with inconsistent estimates"""
        estimated = {
            'neel_to_intermediate': 0.20,  # Too low
            'intermediate_to_stripe': 0.80  # Too high
        }
        
        results = validation_module.compare_with_literature(estimated)
        
        # Both should be inconsistent
        assert results['neel_to_intermediate'].is_consistent is False
        assert results['intermediate_to_stripe'].is_consistent is False
    
    def test_compare_with_literature_custom_ranges(self, validation_module):
        """Test literature comparison with custom literature ranges"""
        estimated = {'transition_1': 0.50}
        literature = {'transition_1': (0.45, 0.55)}
        
        results = validation_module.compare_with_literature(estimated, literature)
        
        assert results['transition_1'].is_consistent is True
        assert results['transition_1'].literature_range == (0.45, 0.55)
    
    def test_compare_with_literature_unknown_transition(self, validation_module):
        """Test literature comparison with unknown transition"""
        estimated = {'unknown_transition': 0.50}
        
        results = validation_module.compare_with_literature(estimated)
        
        # Should handle gracefully
        assert 'unknown_transition' in results
        assert results['unknown_transition'].is_consistent is True  # Can't validate
    
    def test_validate_all_success(self, validation_module, sample_correlations, 
                                  sample_observables_long, sample_latent_reps):
        """Test comprehensive validation with all checks passing"""
        estimated_critical_points = {
            'neel_to_intermediate': 0.42,
            'intermediate_to_stripe': 0.58
        }
        
        results = validation_module.validate_all(
            sample_correlations,
            sample_observables_long,
            sample_latent_reps,
            estimated_critical_points
        )
        
        assert 'neel_phase' in results
        assert 'stripe_phase' in results
        assert 'phase_separation' in results
        assert 'literature_comparison' in results
        assert 'overall_valid' in results
        
        # All individual checks should pass
        assert results['neel_phase'].is_valid is True
        assert results['stripe_phase'].is_valid is True
        assert results['phase_separation'].is_well_separated is True
        assert results['overall_valid'] is True
    
    def test_validate_all_without_critical_points(self, validation_module, sample_correlations,
                                                  sample_observables_long, sample_latent_reps):
        """Test comprehensive validation without critical point comparison"""
        results = validation_module.validate_all(
            sample_correlations,
            sample_observables_long,
            sample_latent_reps
        )
        
        assert 'neel_phase' in results
        assert 'stripe_phase' in results
        assert 'phase_separation' in results
        assert 'literature_comparison' not in results
        assert 'overall_valid' in results
    
    def test_validate_all_partial_failure(self, validation_module, sample_observables_long, sample_latent_reps):
        """Test comprehensive validation with some checks failing"""
        # Weak correlations
        weak_correlations = pd.DataFrame({
            'staggered_mag': [0.3, 0.2, 0.1],
            'stripe_order': [0.2, 0.3, 0.1],
            'energy': [-0.4, -0.3, -0.2]
        }, index=['z_0', 'z_1', 'z_2'])
        
        results = validation_module.validate_all(
            weak_correlations,
            sample_observables_long,
            sample_latent_reps
        )
        
        # Overall validation should fail
        assert results['overall_valid'] is False
    
    def test_custom_thresholds(self):
        """Test ValidationModule with custom thresholds"""
        module = ValidationModule()
        module.correlation_threshold = 0.9
        module.silhouette_threshold = 0.7
        
        assert module.correlation_threshold == 0.9
        assert module.silhouette_threshold == 0.7
    
    def test_phase_validation_result_dataclass(self):
        """Test PhaseValidationResult dataclass"""
        result = PhaseValidationResult(
            phase_name='Test',
            is_valid=True,
            max_correlation=0.85,
            dominant_latent_dim='z_0',
            dominant_observable='test_obs',
            details={'key': 'value'}
        )
        
        assert result.phase_name == 'Test'
        assert result.is_valid is True
        assert result.max_correlation == 0.85
        assert result.details['key'] == 'value'
    
    def test_phase_separation_result_dataclass(self):
        """Test PhaseSeparationResult dataclass"""
        result = PhaseSeparationResult(
            silhouette_score=0.75,
            is_well_separated=True,
            n_clusters=2,
            cluster_labels=np.array([0, 0, 1, 1]),
            threshold=0.5
        )
        
        assert result.silhouette_score == 0.75
        assert result.is_well_separated is True
        assert result.n_clusters == 2
        assert len(result.cluster_labels) == 4
    
    def test_literature_comparison_result_dataclass(self):
        """Test LiteratureComparisonResult dataclass"""
        result = LiteratureComparisonResult(
            estimated_critical_point=0.42,
            literature_range=(0.38, 0.45),
            is_consistent=True,
            deviation=0.015,
            source='Test'
        )
        
        assert result.estimated_critical_point == 0.42
        assert result.literature_range == (0.38, 0.45)
        assert result.is_consistent is True
        assert result.deviation == 0.015


class TestValidationEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_correlations(self):
        """Test with empty correlation matrix"""
        module = ValidationModule()
        empty_corr = pd.DataFrame()
        empty_obs = pd.DataFrame()
        
        result = module.validate_neel_phase(empty_corr, empty_obs)
        assert result.is_valid is False
    
    def test_nan_correlations(self):
        """Test with NaN values in correlations"""
        module = ValidationModule()
        
        corr = pd.DataFrame({
            'staggered_mag': [np.nan, np.nan, np.nan],
            'stripe_order': [0.5, 0.6, 0.7]
        }, index=['z_0', 'z_1', 'z_2'])
        
        obs = pd.DataFrame([
            {'j2_j1': 0.3, 'L': 4, 'observable_name': 'staggered_mag', 'value': 0.4}
        ])
        
        result = module.validate_neel_phase(corr, obs)
        assert result.is_valid is False
    
    def test_single_data_point(self):
        """Test with single data point"""
        module = ValidationModule()
        
        latent_reps = {(0.3, 4): np.array([1.0, 2.0, 3.0])}
        obs = pd.DataFrame([
            {'j2_j1': 0.3, 'L': 4, 'observable_name': 'staggered_mag', 'value': 0.4}
        ])
        
        result = module.validate_phase_separation(latent_reps, obs)
        # Should handle gracefully
        assert isinstance(result, PhaseSeparationResult)
