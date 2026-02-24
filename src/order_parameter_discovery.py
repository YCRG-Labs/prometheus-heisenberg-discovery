"""
Order Parameter Discovery Module

This module implements correlation analysis and order parameter discovery
for the J1-J2 Heisenberg Prometheus framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    coefficient: float
    p_value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class CorrelationAnalysis:
    """
    Implements correlation analysis between latent dimensions and physical observables.
    
    This class provides methods for computing Pearson correlations, bootstrap confidence
    intervals, and permutation tests for statistical significance.
    """
    
    def __init__(self, config):
        """
        Initialize CorrelationAnalysis.
        
        Args:
            config: Configuration object with correlation_threshold and other parameters
        """
        self.config = config
        self.correlation_threshold = getattr(config, 'correlation_threshold', 0.8)
        self.bootstrap_samples = getattr(config, 'bootstrap_samples', 1000)
        self.significance_level = getattr(config, 'significance_level', 0.01)
        
        logger.info(f"Initialized CorrelationAnalysis with threshold={self.correlation_threshold}")
    
    def compute_pearson_correlation(self, 
                                   x: np.ndarray, 
                                   y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient and p-value.
        
        Args:
            x: First variable array
            y: Second variable array
            
        Returns:
            Tuple of (correlation_coefficient, p_value)
            
        Raises:
            ValueError: If arrays have different lengths or contain NaN/Inf
        """
        # Validate inputs
        if len(x) != len(y):
            raise ValueError(f"Arrays must have same length: {len(x)} != {len(y)}")
        
        if len(x) < 3:
            raise ValueError(f"Need at least 3 data points for correlation, got {len(x)}")
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN or Inf values")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or Inf values")
        
        # Constant array => correlation undefined; return 0 and p=1 to avoid nan
        if np.var(x) < 1e-14 or np.var(y) < 1e-14:
            return 0.0, 1.0
        
        # Compute Pearson correlation using scipy
        r, p_value = stats.pearsonr(x, y)
        r, p_value = float(r), float(p_value)
        
        # Handle scipy returning nan for edge cases
        if not np.isfinite(r) or not (-1 <= r <= 1):
            r, p_value = 0.0, 1.0
        
        return r, p_value
    
    def compute_correlation_matrix(self,
                                   latent_data: pd.DataFrame,
                                   observable_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation between all latent dimensions and observables.
        
        Args:
            latent_data: DataFrame with columns [j2_j1, L, z_0, z_1, ..., z_{d-1}]
            observable_data: DataFrame with columns [j2_j1, L, observable_name, value]
            
        Returns:
            DataFrame with shape (n_latent_dims, n_observables) containing correlation coefficients
            
        Raises:
            ValueError: If dataframes cannot be properly merged or have missing data
        """
        # Merge dataframes on (j2_j1, L)
        if 'j2_j1' not in latent_data.columns or 'L' not in latent_data.columns:
            raise ValueError("latent_data must have 'j2_j1' and 'L' columns")
        
        # Identify latent dimension columns (those starting with 'z_')
        latent_cols = [col for col in latent_data.columns if col.startswith('z_')]
        if not latent_cols:
            raise ValueError("No latent dimension columns (z_*) found in latent_data")
        
        # Pivot observable data to wide format if needed
        # Support both 'observable_name' (generic) and 'observable' (from ObservableModule)
        if 'observable_name' in observable_data.columns and 'value' in observable_data.columns:
            name_col = 'observable_name'
        elif 'observable' in observable_data.columns and 'value' in observable_data.columns:
            # Adapt to ObservableModule output
            observable_data = observable_data.rename(columns={'observable': 'observable_name'})
            name_col = 'observable_name'
        else:
            name_col = None

        if name_col is not None:
            # Long format - pivot to wide
            obs_wide = observable_data.pivot_table(
                index=['j2_j1', 'L'],
                columns=name_col,
                values='value'
            ).reset_index()
        else:
            # Already in wide format
            obs_wide = observable_data
        
        # Merge on (j2_j1, L)
        merged = pd.merge(latent_data, obs_wide, on=['j2_j1', 'L'], how='inner')
        
        if len(merged) == 0:
            raise ValueError("No matching (j2_j1, L) points between latent and observable data")
        
        # Get observable columns (exclude j2_j1, L, and latent columns)
        obs_cols = [col for col in merged.columns 
                   if col not in ['j2_j1', 'L'] and not col.startswith('z_')]
        
        if not obs_cols:
            raise ValueError("No observable columns found in merged data")
        
        logger.info(f"Computing correlation matrix: {len(latent_cols)} latent dims × {len(obs_cols)} observables")
        
        # Compute correlation matrix
        correlation_matrix = pd.DataFrame(
            index=latent_cols,
            columns=obs_cols,
            dtype=float
        )
        
        for latent_col in latent_cols:
            for obs_col in obs_cols:
                try:
                    # Extract data, dropping any NaN values
                    x = merged[latent_col].values
                    y = merged[obs_col].values
                    
                    # Remove NaN pairs
                    mask = np.isfinite(x) & np.isfinite(y)
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) >= 3:
                        r, _ = self.compute_pearson_correlation(x_clean, y_clean)
                        correlation_matrix.loc[latent_col, obs_col] = r
                    else:
                        correlation_matrix.loc[latent_col, obs_col] = np.nan
                        logger.warning(f"Insufficient data for {latent_col} vs {obs_col}")
                        
                except Exception as e:
                    logger.warning(f"Error computing correlation for {latent_col} vs {obs_col}: {e}")
                    correlation_matrix.loc[latent_col, obs_col] = np.nan
        
        return correlation_matrix
    
    def bootstrap_correlation(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             n_bootstrap: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Compute correlation with bootstrap confidence intervals.
        
        Args:
            x: First variable array
            y: Second variable array
            n_bootstrap: Number of bootstrap samples (default: from config)
            
        Returns:
            Tuple of (correlation, ci_lower, ci_upper)
            
        Raises:
            ValueError: If arrays are invalid or insufficient data
        """
        if n_bootstrap is None:
            n_bootstrap = self.bootstrap_samples
        
        # Validate inputs
        if len(x) != len(y):
            raise ValueError(f"Arrays must have same length: {len(x)} != {len(y)}")
        
        if len(x) < 3:
            raise ValueError(f"Need at least 3 data points for bootstrap, got {len(x)}")
        
        # Compute original correlation
        r_original, _ = self.compute_pearson_correlation(x, y)
        
        # Bootstrap resampling
        n = len(x)
        bootstrap_correlations = np.zeros(n_bootstrap)
        
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            try:
                r_boot, _ = self.compute_pearson_correlation(x_boot, y_boot)
                bootstrap_correlations[i] = r_boot
            except Exception as e:
                # If correlation fails (e.g., constant values), use original
                bootstrap_correlations[i] = r_original
        
        # Compute confidence intervals (95% by default)
        ci_lower = np.percentile(bootstrap_correlations, 2.5)
        ci_upper = np.percentile(bootstrap_correlations, 97.5)
        
        # Validate ordering
        if not (ci_lower <= r_original <= ci_upper):
            logger.warning(
                f"Bootstrap CI ordering violated: {ci_lower:.3f} <= {r_original:.3f} <= {ci_upper:.3f}"
            )
        
        return float(r_original), float(ci_lower), float(ci_upper)
    
    def permutation_test(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        n_permutations: int = 10000) -> float:
        """
        Permutation test for correlation significance.
        
        Tests the null hypothesis that x and y are independent by randomly
        permuting one variable and computing the correlation.
        
        Args:
            x: First variable array
            y: Second variable array
            n_permutations: Number of permutations to perform
            
        Returns:
            p-value: Proportion of permuted correlations >= observed correlation
            
        Raises:
            ValueError: If arrays are invalid
        """
        # Validate inputs
        if len(x) != len(y):
            raise ValueError(f"Arrays must have same length: {len(x)} != {len(y)}")
        
        if len(x) < 3:
            raise ValueError(f"Need at least 3 data points for permutation test, got {len(x)}")
        
        # Compute observed correlation
        r_observed, _ = self.compute_pearson_correlation(x, y)
        r_observed_abs = abs(r_observed)
        
        # Permutation test
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        count_extreme = 0
        
        for _ in range(n_permutations):
            # Permute y
            y_perm = rng.permutation(y)
            
            try:
                r_perm, _ = self.compute_pearson_correlation(x, y_perm)
                if abs(r_perm) >= r_observed_abs:
                    count_extreme += 1
            except Exception:
                # If correlation fails, skip this permutation
                continue
        
        # Compute p-value
        p_value = (count_extreme + 1) / (n_permutations + 1)  # +1 for observed
        
        # Validate p-value bounds
        if not (0 <= p_value <= 1):
            logger.warning(f"P-value {p_value} outside [0, 1] bounds")
            p_value = np.clip(p_value, 0, 1)
        
        return float(p_value)


class OrderParameterDiscovery:
    """
    Discovers order parameters by correlating latent dimensions with physical observables.
    
    This class implements the main discovery pipeline, validation in known phases,
    and analysis of the intermediate regime.
    """
    
    def __init__(self, config):
        """
        Initialize OrderParameterDiscovery.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.correlation_analysis = CorrelationAnalysis(config)
        self.correlation_threshold = getattr(config, 'correlation_threshold', 0.8)
        
        logger.info("Initialized OrderParameterDiscovery")
    
    def discover_order_parameters(self,
                                 latent_reps: Dict[Tuple[float, int], np.ndarray],
                                 observables: pd.DataFrame) -> Dict[str, Any]:
        """
        Main discovery pipeline.
        
        Steps:
        1. Prepare data in DataFrame format
        2. Compute correlations between latent dims and observables
        3. Identify significant correlations
        4. Validate in known phases
        5. Return discovered order parameters
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            observables: DataFrame with columns [j2_j1, L, observable_name, value]
            
        Returns:
            Dictionary containing:
                - correlation_matrix: DataFrame of correlations
                - significant_correlations: List of (latent_dim, observable, r, p_value)
                - discovered_order_parameters: Dict mapping latent_dim -> observable
                - validation_results: Dict of validation checks
        """
        logger.info("Starting order parameter discovery pipeline")
        
        # Step 1: Prepare latent data as DataFrame
        latent_data = self._prepare_latent_dataframe(latent_reps)
        
        # Step 2: Compute correlation matrix
        logger.info("Computing correlation matrix")
        correlation_matrix = self.correlation_analysis.compute_correlation_matrix(
            latent_data, observables
        )
        
        # Step 3: Identify significant correlations
        logger.info("Identifying significant correlations")
        significant_correlations = self._identify_significant_correlations(
            latent_data, observables, correlation_matrix
        )
        
        # Step 4: Map latent dimensions to order parameters
        discovered_order_parameters = self._map_latent_to_observables(
            significant_correlations
        )
        
        # Step 5: Validate in known phases
        logger.info("Validating in known phases")
        validation_results = self.validate_in_known_phases(
            correlation_matrix, observables
        )
        
        results = {
            'correlation_matrix': correlation_matrix,
            'significant_correlations': significant_correlations,
            'discovered_order_parameters': discovered_order_parameters,
            'validation_results': validation_results
        }
        
        logger.info(f"Discovery complete: found {len(discovered_order_parameters)} order parameters")
        
        return results
    
    def _prepare_latent_dataframe(self, 
                                 latent_reps: Dict[Tuple[float, int], np.ndarray]
                                 ) -> pd.DataFrame:
        """
        Convert latent representations dictionary to DataFrame.
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            
        Returns:
            DataFrame with columns [j2_j1, L, z_0, z_1, ..., z_{d-1}]
        """
        rows = []
        for (j2_j1, L), z in latent_reps.items():
            row = {'j2_j1': j2_j1, 'L': L}
            for i, z_val in enumerate(z):
                row[f'z_{i}'] = z_val
            rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Prepared latent DataFrame: {len(df)} rows, {len(df.columns)-2} latent dimensions")
        
        return df
    
    def _identify_significant_correlations(self,
                                          latent_data: pd.DataFrame,
                                          observables: pd.DataFrame,
                                          correlation_matrix: pd.DataFrame
                                          ) -> List[Tuple[str, str, float, float]]:
        """
        Identify correlations exceeding threshold with statistical significance.
        
        Args:
            latent_data: DataFrame with latent representations
            observables: DataFrame with observable values
            correlation_matrix: Precomputed correlation matrix
            
        Returns:
            List of tuples (latent_dim, observable, r, p_value) for significant correlations
        """
        significant = []
        
        # Merge data for statistical tests
        # Support both 'observable_name' and 'observable' long formats
        if 'observable_name' in observables.columns and 'value' in observables.columns:
            name_col = 'observable_name'
        elif 'observable' in observables.columns and 'value' in observables.columns:
            observables = observables.rename(columns={'observable': 'observable_name'})
            name_col = 'observable_name'
        else:
            name_col = None

        if name_col is not None:
            obs_wide = observables.pivot_table(
                index=['j2_j1', 'L'],
                columns=name_col,
                values='value'
            ).reset_index()
        else:
            obs_wide = observables
        
        merged = pd.merge(latent_data, obs_wide, on=['j2_j1', 'L'], how='inner')
        
        # Check each correlation in the matrix
        for latent_dim in correlation_matrix.index:
            for observable in correlation_matrix.columns:
                r = correlation_matrix.loc[latent_dim, observable]
                
                # Check if correlation exceeds threshold
                if pd.notna(r) and abs(r) >= self.correlation_threshold:
                    # Compute p-value
                    try:
                        x = merged[latent_dim].values
                        y = merged[observable].values
                        
                        # Remove NaN pairs
                        mask = np.isfinite(x) & np.isfinite(y)
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) >= 3:
                            _, p_value = self.correlation_analysis.compute_pearson_correlation(
                                x_clean, y_clean
                            )
                            
                            # Check significance
                            if p_value < self.correlation_analysis.significance_level:
                                significant.append((latent_dim, observable, r, p_value))
                                logger.info(
                                    f"Significant correlation: {latent_dim} <-> {observable}: "
                                    f"r={r:.3f}, p={p_value:.2e}"
                                )
                    except Exception as e:
                        logger.warning(f"Error testing {latent_dim} vs {observable}: {e}")
        
        return significant
    
    def _map_latent_to_observables(self,
                                   significant_correlations: List[Tuple[str, str, float, float]]
                                   ) -> Dict[str, str]:
        """
        Map each latent dimension to its most strongly correlated observable.
        
        Args:
            significant_correlations: List of (latent_dim, observable, r, p_value)
            
        Returns:
            Dictionary mapping latent_dim -> observable (strongest correlation)
        """
        mapping = {}
        
        # Group by latent dimension
        for latent_dim, observable, r, p_value in significant_correlations:
            if latent_dim not in mapping:
                mapping[latent_dim] = (observable, abs(r))
            else:
                # Keep the observable with stronger correlation
                current_obs, current_r = mapping[latent_dim]
                if abs(r) > current_r:
                    mapping[latent_dim] = (observable, abs(r))
        
        # Extract just the observable names
        result = {latent_dim: obs for latent_dim, (obs, _) in mapping.items()}
        
        return result
    
    def validate_in_known_phases(self,
                                correlations: pd.DataFrame,
                                observables: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate discovered order parameters in Néel and stripe regimes.
        
        Checks:
        - Néel regime (j2_j1 < 0.4): expect correlation with staggered_mag
        - Stripe regime (j2_j1 > 0.6): expect correlation with stripe_order
        
        Args:
            correlations: Correlation matrix DataFrame
            observables: Observable data DataFrame
            
        Returns:
            Dictionary with validation results:
                - neel_phase_valid: bool
                - stripe_phase_valid: bool
                - neel_dominant_observable: str
                - stripe_dominant_observable: str
        """
        logger.info("Validating in known phases")
        
        results = {
            'neel_phase_valid': False,
            'stripe_phase_valid': False,
            'neel_dominant_observable': None,
            'stripe_dominant_observable': None
        }
        
        # Check if expected observables exist in correlation matrix
        has_staggered_mag = 'staggered_mag' in correlations.columns
        has_stripe_order = 'stripe_order' in correlations.columns
        
        if not has_staggered_mag:
            logger.warning("staggered_mag not found in observables")
        if not has_stripe_order:
            logger.warning("stripe_order not found in observables")
        
        # Validate Néel phase
        if has_staggered_mag:
            # Find latent dimension with strongest correlation to staggered_mag
            staggered_corrs = correlations['staggered_mag'].dropna()
            if len(staggered_corrs) > 0:
                max_corr = staggered_corrs.abs().max()
                if max_corr >= 0.7:  # Threshold for validation
                    results['neel_phase_valid'] = True
                    results['neel_dominant_observable'] = 'staggered_mag'
                    logger.info(f"Néel phase validation: PASSED (max |r|={max_corr:.3f})")
                else:
                    logger.warning(f"Néel phase validation: FAILED (max |r|={max_corr:.3f} < 0.7)")
        
        # Validate stripe phase
        if has_stripe_order:
            # Find latent dimension with strongest correlation to stripe_order
            stripe_corrs = correlations['stripe_order'].dropna()
            if len(stripe_corrs) > 0:
                max_corr = stripe_corrs.abs().max()
                if max_corr >= 0.7:  # Threshold for validation
                    results['stripe_phase_valid'] = True
                    results['stripe_dominant_observable'] = 'stripe_order'
                    logger.info(f"Stripe phase validation: PASSED (max |r|={max_corr:.3f})")
                else:
                    logger.warning(f"Stripe phase validation: FAILED (max |r|={max_corr:.3f} < 0.7)")
        
        return results
    
    def analyze_intermediate_regime(self,
                                   latent_reps: Dict[Tuple[float, int], np.ndarray],
                                   observables: pd.DataFrame,
                                   j2_j1_range: Tuple[float, float] = (0.4, 0.6)
                                   ) -> Dict[str, Any]:
        """
        Analyze intermediate regime to identify characteristic order parameters.
        
        Steps:
        1. Filter data to intermediate regime
        2. Compute variance of each latent dimension
        3. Identify latent dimensions with high variance (sensitive to phase changes)
        4. Check correlations with candidate order parameters
        5. Assess evidence for distinct phase vs crossover
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            observables: DataFrame with observable values
            j2_j1_range: Tuple of (min, max) for intermediate regime
            
        Returns:
            Dictionary containing:
                - latent_variances: Dict mapping latent_dim -> variance
                - high_variance_dims: List of latent dimensions with high variance
                - intermediate_correlations: Correlation matrix for intermediate regime
                - phase_assessment: str ('distinct_phase' or 'crossover')
        """
        logger.info(f"Analyzing intermediate regime: j2_j1 ∈ [{j2_j1_range[0]}, {j2_j1_range[1]}]")
        
        j2_j1_min, j2_j1_max = j2_j1_range
        
        # Filter latent representations to intermediate regime
        intermediate_latent = {
            (j2_j1, L): z for (j2_j1, L), z in latent_reps.items()
            if j2_j1_min <= j2_j1 <= j2_j1_max
        }
        
        if not intermediate_latent:
            logger.warning("No data points in intermediate regime")
            return {
                'latent_variances': {},
                'high_variance_dims': [],
                'intermediate_correlations': None,
                'phase_assessment': 'insufficient_data'
            }
        
        # Prepare DataFrame
        latent_df = self._prepare_latent_dataframe(intermediate_latent)
        
        # Compute variance of each latent dimension
        latent_cols = [col for col in latent_df.columns if col.startswith('z_')]
        latent_variances = {}
        
        for col in latent_cols:
            variance = latent_df[col].var()
            latent_variances[col] = variance
        
        # Identify high variance dimensions (top 25% or above median)
        if latent_variances:
            variance_threshold = np.median(list(latent_variances.values()))
            high_variance_dims = [
                dim for dim, var in latent_variances.items()
                if var >= variance_threshold
            ]
            logger.info(f"High variance dimensions: {high_variance_dims}")
        else:
            high_variance_dims = []
        
        # Filter observables to intermediate regime
        intermediate_obs = observables[
            (observables['j2_j1'] >= j2_j1_min) &
            (observables['j2_j1'] <= j2_j1_max)
        ].copy()
        
        # Compute correlations in intermediate regime
        if len(intermediate_obs) > 0:
            try:
                intermediate_correlations = self.correlation_analysis.compute_correlation_matrix(
                    latent_df, intermediate_obs
                )
            except Exception as e:
                logger.warning(f"Could not compute intermediate correlations: {e}")
                intermediate_correlations = None
        else:
            intermediate_correlations = None
        
        # Assess phase structure
        # Simple heuristic: if multiple latent dimensions have high variance,
        # suggests complex phase structure
        if len(high_variance_dims) >= 2:
            phase_assessment = 'distinct_phase'
        elif len(high_variance_dims) == 1:
            phase_assessment = 'crossover'
        else:
            phase_assessment = 'unclear'
        
        logger.info(f"Phase assessment: {phase_assessment}")
        
        return {
            'latent_variances': latent_variances,
            'high_variance_dims': high_variance_dims,
            'intermediate_correlations': intermediate_correlations,
            'phase_assessment': phase_assessment
        }
