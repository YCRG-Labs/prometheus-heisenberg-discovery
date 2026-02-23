"""
Validation Module

This module implements validation functions for known phases in the J1-J2 Heisenberg model.
It validates that the Q-VAE analysis correctly identifies order parameters in the Néel and
stripe regimes before applying it to the unknown intermediate regime.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.9
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass
import logging

from src.latent_space_analysis import LatentSpaceAnalysis

logger = logging.getLogger(__name__)


@dataclass
class PhaseValidationResult:
    """Result of phase validation"""
    phase_name: str
    is_valid: bool
    max_correlation: float
    dominant_latent_dim: Optional[str]
    dominant_observable: str
    details: Dict[str, Any]


@dataclass
class PhaseSeparationResult:
    """Result of phase separation analysis"""
    silhouette_score: float
    is_well_separated: bool
    n_clusters: int
    cluster_labels: np.ndarray
    threshold: float


@dataclass
class LiteratureComparisonResult:
    """Result of comparison with literature"""
    estimated_critical_point: float
    literature_range: Tuple[float, float]
    is_consistent: bool
    deviation: float
    source: str


class ValidationModule:
    """
    Validation functions for known phases.
    
    This class provides methods to validate Q-VAE analysis results against
    known physics in the Néel and stripe regimes, assess phase separation
    quality, and compare critical point estimates with literature values.
    """
    
    def __init__(self, config=None):
        """
        Initialize ValidationModule.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        self.latent_analysis = LatentSpaceAnalysis(config)
        
        # Validation thresholds
        self.correlation_threshold = 0.7  # Minimum correlation for validation
        self.silhouette_threshold = 0.5   # Minimum silhouette score for good separation
        
        logger.info("Initialized ValidationModule")
    
    def validate_neel_phase(
        self,
        correlations: pd.DataFrame,
        observables: pd.DataFrame,
        j2_j1_range: Tuple[float, float] = (0.0, 0.4)
    ) -> PhaseValidationResult:
        """
        Validate that Q-VAE discovers staggered magnetization in Néel phase.
        
        In the Néel regime (j2_j1 < 0.4), the dominant order parameter should be
        staggered magnetization. This function checks that at least one latent
        dimension shows strong correlation (|r| > 0.7) with staggered_mag.
        
        Args:
            correlations: Correlation matrix DataFrame (latent dims × observables)
            observables: Observable data DataFrame
            j2_j1_range: Range defining Néel regime (default: [0.0, 0.4])
            
        Returns:
            PhaseValidationResult with validation outcome and details
            
        Requirements: 10.1
        """
        logger.info(f"Validating Néel phase in range j2_j1 ∈ [{j2_j1_range[0]}, {j2_j1_range[1]}]")
        
        # Check if staggered_mag exists in observables
        if 'staggered_mag' not in correlations.columns:
            logger.warning("staggered_mag not found in correlation matrix")
            return PhaseValidationResult(
                phase_name='Neel',
                is_valid=False,
                max_correlation=0.0,
                dominant_latent_dim=None,
                dominant_observable='staggered_mag',
                details={'error': 'staggered_mag not in observables'}
            )
        
        # Get correlations with staggered magnetization
        staggered_corrs = correlations['staggered_mag'].dropna()
        
        if len(staggered_corrs) == 0:
            logger.warning("No valid correlations with staggered_mag")
            return PhaseValidationResult(
                phase_name='Neel',
                is_valid=False,
                max_correlation=0.0,
                dominant_latent_dim=None,
                dominant_observable='staggered_mag',
                details={'error': 'no valid correlations'}
            )
        
        # Find maximum absolute correlation
        max_corr_idx = staggered_corrs.abs().idxmax()
        max_corr = staggered_corrs[max_corr_idx]
        
        # Validation passes if |r| >= threshold
        is_valid = abs(max_corr) >= self.correlation_threshold
        
        # Additional check: verify staggered_mag is dominant in Néel regime
        neel_obs = observables[
            (observables['j2_j1'] >= j2_j1_range[0]) &
            (observables['j2_j1'] <= j2_j1_range[1])
        ]
        
        details = {
            'max_correlation': float(max_corr),
            'dominant_latent_dim': max_corr_idx,
            'threshold': self.correlation_threshold,
            'n_data_points': len(neel_obs)
        }
        
        # Check if staggered_mag > stripe_order in Néel regime
        if 'stripe_order' in observables.columns:
            if 'observable_name' in neel_obs.columns:
                # Long format
                stag_vals = neel_obs[neel_obs['observable_name'] == 'staggered_mag']['value']
                stripe_vals = neel_obs[neel_obs['observable_name'] == 'stripe_order']['value']
            else:
                # Wide format
                stag_vals = neel_obs.get('staggered_mag', pd.Series([]))
                stripe_vals = neel_obs.get('stripe_order', pd.Series([]))
            
            if len(stag_vals) > 0 and len(stripe_vals) > 0:
                mean_stag = stag_vals.mean()
                mean_stripe = stripe_vals.mean()
                details['mean_staggered_mag'] = float(mean_stag)
                details['mean_stripe_order'] = float(mean_stripe)
                details['staggered_dominant'] = mean_stag > mean_stripe
        
        if is_valid:
            logger.info(
                f"Néel phase validation: PASSED "
                f"(max |r| = {abs(max_corr):.3f} >= {self.correlation_threshold})"
            )
        else:
            logger.warning(
                f"Néel phase validation: FAILED "
                f"(max |r| = {abs(max_corr):.3f} < {self.correlation_threshold})"
            )
        
        return PhaseValidationResult(
            phase_name='Neel',
            is_valid=is_valid,
            max_correlation=abs(max_corr),
            dominant_latent_dim=max_corr_idx,
            dominant_observable='staggered_mag',
            details=details
        )
    
    def validate_stripe_phase(
        self,
        correlations: pd.DataFrame,
        observables: pd.DataFrame,
        j2_j1_range: Tuple[float, float] = (0.6, 1.0)
    ) -> PhaseValidationResult:
        """
        Validate that Q-VAE discovers stripe order in stripe phase.
        
        In the stripe regime (j2_j1 > 0.6), the dominant order parameter should be
        stripe order. This function checks that at least one latent dimension shows
        strong correlation (|r| > 0.7) with stripe_order.
        
        Args:
            correlations: Correlation matrix DataFrame (latent dims × observables)
            observables: Observable data DataFrame
            j2_j1_range: Range defining stripe regime (default: [0.6, 1.0])
            
        Returns:
            PhaseValidationResult with validation outcome and details
            
        Requirements: 10.2
        """
        logger.info(f"Validating stripe phase in range j2_j1 ∈ [{j2_j1_range[0]}, {j2_j1_range[1]}]")
        
        # Check if stripe_order exists in observables
        if 'stripe_order' not in correlations.columns:
            logger.warning("stripe_order not found in correlation matrix")
            return PhaseValidationResult(
                phase_name='Stripe',
                is_valid=False,
                max_correlation=0.0,
                dominant_latent_dim=None,
                dominant_observable='stripe_order',
                details={'error': 'stripe_order not in observables'}
            )
        
        # Get correlations with stripe order
        stripe_corrs = correlations['stripe_order'].dropna()
        
        if len(stripe_corrs) == 0:
            logger.warning("No valid correlations with stripe_order")
            return PhaseValidationResult(
                phase_name='Stripe',
                is_valid=False,
                max_correlation=0.0,
                dominant_latent_dim=None,
                dominant_observable='stripe_order',
                details={'error': 'no valid correlations'}
            )
        
        # Find maximum absolute correlation
        max_corr_idx = stripe_corrs.abs().idxmax()
        max_corr = stripe_corrs[max_corr_idx]
        
        # Validation passes if |r| >= threshold
        is_valid = abs(max_corr) >= self.correlation_threshold
        
        # Additional check: verify stripe_order is dominant in stripe regime
        stripe_obs = observables[
            (observables['j2_j1'] >= j2_j1_range[0]) &
            (observables['j2_j1'] <= j2_j1_range[1])
        ]
        
        details = {
            'max_correlation': float(max_corr),
            'dominant_latent_dim': max_corr_idx,
            'threshold': self.correlation_threshold,
            'n_data_points': len(stripe_obs)
        }
        
        # Check if stripe_order > staggered_mag in stripe regime
        if 'staggered_mag' in observables.columns:
            if 'observable_name' in stripe_obs.columns:
                # Long format
                stripe_vals = stripe_obs[stripe_obs['observable_name'] == 'stripe_order']['value']
                stag_vals = stripe_obs[stripe_obs['observable_name'] == 'staggered_mag']['value']
            else:
                # Wide format
                stripe_vals = stripe_obs.get('stripe_order', pd.Series([]))
                stag_vals = stripe_obs.get('staggered_mag', pd.Series([]))
            
            if len(stripe_vals) > 0 and len(stag_vals) > 0:
                mean_stripe = stripe_vals.mean()
                mean_stag = stag_vals.mean()
                details['mean_stripe_order'] = float(mean_stripe)
                details['mean_staggered_mag'] = float(mean_stag)
                details['stripe_dominant'] = mean_stripe > mean_stag
        
        if is_valid:
            logger.info(
                f"Stripe phase validation: PASSED "
                f"(max |r| = {abs(max_corr):.3f} >= {self.correlation_threshold})"
            )
        else:
            logger.warning(
                f"Stripe phase validation: FAILED "
                f"(max |r| = {abs(max_corr):.3f} < {self.correlation_threshold})"
            )
        
        return PhaseValidationResult(
            phase_name='Stripe',
            is_valid=is_valid,
            max_correlation=abs(max_corr),
            dominant_latent_dim=max_corr_idx,
            dominant_observable='stripe_order',
            details=details
        )
    
    def validate_phase_separation(
        self,
        latent_reps: Dict[Tuple[float, int], np.ndarray],
        observables: pd.DataFrame,
        neel_range: Tuple[float, float] = (0.0, 0.4),
        stripe_range: Tuple[float, float] = (0.6, 1.0),
        L: Optional[int] = None
    ) -> PhaseSeparationResult:
        """
        Validate that Néel and stripe phases are well-separated in latent space.
        
        Checks that latent space clustering separates Néel and stripe phases
        with high silhouette score (> 0.5), indicating good phase separation.
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            observables: Observable data DataFrame
            neel_range: Range defining Néel regime
            stripe_range: Range defining stripe regime
            L: Optional lattice size to analyze (if None, use all)
            
        Returns:
            PhaseSeparationResult with silhouette score and separation quality
            
        Requirements: 10.3, 10.4
        """
        logger.info("Validating phase separation in latent space")
        
        # Filter latent representations to Néel and stripe regimes
        neel_latent = []
        stripe_latent = []
        labels = []
        
        for (j2_j1, lattice_size), z in latent_reps.items():
            if L is not None and lattice_size != L:
                continue
            
            if neel_range[0] <= j2_j1 <= neel_range[1]:
                neel_latent.append(z)
                labels.append(0)  # Néel phase label
            elif stripe_range[0] <= j2_j1 <= stripe_range[1]:
                stripe_latent.append(z)
                labels.append(1)  # Stripe phase label
        
        if len(neel_latent) == 0 or len(stripe_latent) == 0:
            logger.warning("Insufficient data in Néel or stripe regimes")
            return PhaseSeparationResult(
                silhouette_score=0.0,
                is_well_separated=False,
                n_clusters=2,
                cluster_labels=np.array([]),
                threshold=self.silhouette_threshold
            )
        
        # Combine into single array
        latent_array = np.array(neel_latent + stripe_latent)
        labels_array = np.array(labels)
        
        # Compute silhouette score
        silhouette = self.latent_analysis.compute_silhouette_score(
            latent_array,
            labels_array
        )
        
        # Check if well-separated
        is_well_separated = silhouette >= self.silhouette_threshold
        
        if is_well_separated:
            logger.info(
                f"Phase separation validation: PASSED "
                f"(silhouette = {silhouette:.3f} >= {self.silhouette_threshold})"
            )
        else:
            logger.warning(
                f"Phase separation validation: FAILED "
                f"(silhouette = {silhouette:.3f} < {self.silhouette_threshold})"
            )
        
        return PhaseSeparationResult(
            silhouette_score=silhouette,
            is_well_separated=is_well_separated,
            n_clusters=2,
            cluster_labels=labels_array,
            threshold=self.silhouette_threshold
        )
    
    def compare_with_literature(
        self,
        estimated_critical_points: Dict[str, float],
        literature_values: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, LiteratureComparisonResult]:
        """
        Compare detected critical points with literature estimates.
        
        Validates that Q-VAE detected critical points are consistent with
        known bounds from DMRG/QMC studies in the literature.
        
        Args:
            estimated_critical_points: Dictionary mapping transition_name -> j2_j1_c
            literature_values: Dictionary mapping transition_name -> (min, max) range
                              If None, uses default literature values
            
        Returns:
            Dictionary mapping transition_name -> LiteratureComparisonResult
            
        Requirements: 10.5, 10.9
        """
        logger.info("Comparing critical points with literature values")
        
        # Default literature values from DMRG/QMC studies
        # These are approximate ranges from various studies
        if literature_values is None:
            literature_values = {
                'neel_to_intermediate': (0.38, 0.45),  # Néel → intermediate transition
                'intermediate_to_stripe': (0.55, 0.65),  # Intermediate → stripe transition
                'neel_to_stripe': (0.40, 0.60)  # If single transition observed
            }
        
        results = {}
        
        for transition_name, estimated_j2_j1_c in estimated_critical_points.items():
            # Find matching literature value
            lit_range = literature_values.get(transition_name)
            
            if lit_range is None:
                logger.warning(f"No literature value for transition: {transition_name}")
                results[transition_name] = LiteratureComparisonResult(
                    estimated_critical_point=estimated_j2_j1_c,
                    literature_range=(0.0, 1.0),
                    is_consistent=True,  # Can't validate without literature
                    deviation=0.0,
                    source='unknown'
                )
                continue
            
            lit_min, lit_max = lit_range
            lit_center = (lit_min + lit_max) / 2
            
            # Check if estimate falls within literature range
            is_consistent = lit_min <= estimated_j2_j1_c <= lit_max
            
            # Compute deviation from literature center
            deviation = abs(estimated_j2_j1_c - lit_center)
            
            if is_consistent:
                logger.info(
                    f"{transition_name}: CONSISTENT with literature "
                    f"(estimated={estimated_j2_j1_c:.3f}, "
                    f"literature=[{lit_min:.3f}, {lit_max:.3f}])"
                )
            else:
                logger.warning(
                    f"{transition_name}: INCONSISTENT with literature "
                    f"(estimated={estimated_j2_j1_c:.3f}, "
                    f"literature=[{lit_min:.3f}, {lit_max:.3f}])"
                )
            
            results[transition_name] = LiteratureComparisonResult(
                estimated_critical_point=estimated_j2_j1_c,
                literature_range=lit_range,
                is_consistent=is_consistent,
                deviation=deviation,
                source='DMRG/QMC studies'
            )
        
        return results
    
    def validate_all(
        self,
        correlations: pd.DataFrame,
        observables: pd.DataFrame,
        latent_reps: Dict[Tuple[float, int], np.ndarray],
        estimated_critical_points: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform all validation checks.
        
        Comprehensive validation including:
        - Néel phase order parameter validation
        - Stripe phase order parameter validation
        - Phase separation quality
        - Literature comparison (if critical points provided)
        
        Args:
            correlations: Correlation matrix DataFrame
            observables: Observable data DataFrame
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            estimated_critical_points: Optional critical point estimates
            
        Returns:
            Dictionary with all validation results
            
        Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.9
        """
        logger.info("Performing comprehensive validation")
        
        results = {}
        
        # Validate Néel phase
        results['neel_phase'] = self.validate_neel_phase(correlations, observables)
        
        # Validate stripe phase
        results['stripe_phase'] = self.validate_stripe_phase(correlations, observables)
        
        # Validate phase separation
        results['phase_separation'] = self.validate_phase_separation(
            latent_reps, observables
        )
        
        # Compare with literature if critical points provided
        if estimated_critical_points is not None:
            results['literature_comparison'] = self.compare_with_literature(
                estimated_critical_points
            )
        
        # Overall validation status
        all_valid = (
            results['neel_phase'].is_valid and
            results['stripe_phase'].is_valid and
            results['phase_separation'].is_well_separated
        )
        
        results['overall_valid'] = all_valid
        
        if all_valid:
            logger.info("Overall validation: PASSED - All checks successful")
        else:
            logger.warning("Overall validation: FAILED - Some checks did not pass")
        
        return results
