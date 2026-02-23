"""
Finite-Size Scaling Module

This module implements finite-size scaling analysis to extract critical exponents
and extrapolate to the thermodynamic limit for the J1-J2 Heisenberg model.

The scaling form is: O(λ, L) = L^(-x_O/ν) * f_O([(λ - λ_c) * L^(1/ν)])
where:
- λ = j2_j1 (frustration ratio)
- λ_c = critical value of j2_j1
- ν = correlation length exponent
- x_O = scaling dimension of observable O
- f_O = universal scaling function
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class FiniteSizeScaling:
    """
    Finite-size scaling analysis for phase transitions.
    
    Extracts critical exponents and critical points by optimizing
    data collapse quality across different system sizes.
    """
    
    def __init__(self, config: Any):
        """
        Initialize finite-size scaling analyzer.
        
        Args:
            config: Configuration object with bootstrap_samples parameter
        """
        self.config = config
        self.bootstrap_samples = getattr(config, 'bootstrap_samples', 1000)
        
    def scaling_ansatz(self,
                      j2_j1: np.ndarray,
                      L: np.ndarray,
                      j2_j1_c: float,
                      nu: float,
                      x_O: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply scaling ansatz to data.
        
        Transforms (j2_j1, L, O) data to scaled coordinates:
        - x_scaled = (j2_j1 - j2_j1_c) * L^(1/ν)
        - y_scaled = O * L^(x_O/ν)
        
        Args:
            j2_j1: Array of frustration ratio values
            L: Array of lattice sizes (same shape as j2_j1)
            j2_j1_c: Critical point estimate
            nu: Correlation length exponent
            x_O: Scaling dimension of observable
            
        Returns:
            Tuple of (x_scaled, y_scaled) arrays
        """
        # Compute scaled x-coordinate
        x_scaled = (j2_j1 - j2_j1_c) * np.power(L, 1.0 / nu)
        
        # Note: y_scaled requires observable values, which are passed separately
        # This method just computes the x-coordinate transformation
        return x_scaled
    
    def collapse_quality(self,
                        x_scaled: np.ndarray,
                        y_scaled: np.ndarray,
                        L_values: np.ndarray) -> float:
        """
        Compute quality metric for scaling collapse.
        
        Uses chi-squared-like metric: measures how well data from different
        system sizes collapse onto a single curve.
        
        Strategy:
        1. Bin data by x_scaled coordinate
        2. For each bin, compute variance across different L values
        3. Sum normalized variances
        
        Args:
            x_scaled: Scaled x-coordinates
            y_scaled: Scaled y-coordinates (observable * L^(x_O/ν))
            L_values: System sizes for each data point
            
        Returns:
            Chi-squared metric (lower is better)
        """
        # Handle edge cases
        if len(x_scaled) < 2:
            return np.inf
            
        # Create bins for x_scaled
        n_bins = min(20, len(x_scaled) // 3)
        if n_bins < 2:
            n_bins = 2
            
        x_min, x_max = np.min(x_scaled), np.max(x_scaled)
        if np.isclose(x_min, x_max):
            return np.inf
            
        bins = np.linspace(x_min, x_max, n_bins + 1)
        
        chi_squared = 0.0
        n_valid_bins = 0
        
        # For each bin, compute variance across different L values
        for i in range(n_bins):
            # Find points in this bin
            if i == n_bins - 1:  # Last bin includes right edge
                mask = (x_scaled >= bins[i]) & (x_scaled <= bins[i + 1])
            else:
                mask = (x_scaled >= bins[i]) & (x_scaled < bins[i + 1])
                
            if np.sum(mask) < 2:
                continue
                
            y_bin = y_scaled[mask]
            L_bin = L_values[mask]
            
            # Check if we have multiple system sizes in this bin
            unique_L = np.unique(L_bin)
            if len(unique_L) < 2:
                continue
                
            # Compute normalized variance of y values in this bin
            mean_y = np.mean(y_bin)
            if np.abs(mean_y) > 1e-10:
                variance = np.var(y_bin) / (mean_y**2 + 1e-10)
            else:
                variance = np.var(y_bin)
                
            chi_squared += variance
            n_valid_bins += 1
            
        # Normalize by number of valid bins
        if n_valid_bins == 0:
            return np.inf
            
        return chi_squared / n_valid_bins

    def optimize_collapse(self,
                         j2_j1: np.ndarray,
                         L: np.ndarray,
                         observable: np.ndarray,
                         j2_j1_c_init: float,
                         nu_bounds: Tuple[float, float] = (0.3, 2.0),
                         x_O_bounds: Tuple[float, float] = (-2.0, 2.0),
                         method: str = 'differential_evolution') -> Dict[str, float]:
        """
        Optimize scaling collapse by varying (j2_j1_c, ν, x_O).
        
        Minimizes the collapse quality metric to find best-fit critical
        parameters and exponents.
        
        Args:
            j2_j1: Array of frustration ratio values
            L: Array of lattice sizes (same shape as j2_j1)
            observable: Array of observable values (same shape as j2_j1)
            j2_j1_c_init: Initial guess for critical point
            nu_bounds: Bounds for correlation length exponent
            x_O_bounds: Bounds for scaling dimension
            method: Optimization method ('differential_evolution' or 'minimize')
            
        Returns:
            Dictionary with keys:
                - j2_j1_c: Optimized critical point
                - nu: Optimized correlation length exponent
                - x_O: Optimized scaling dimension
                - chi_squared: Final collapse quality metric
                - success: Whether optimization converged
        """
        # Validate inputs
        if len(j2_j1) != len(L) or len(j2_j1) != len(observable):
            raise ValueError("Input arrays must have same length")
            
        if len(j2_j1) < 3:
            raise ValueError("Need at least 3 data points for scaling analysis")
            
        # Define objective function
        def objective(params):
            j2_j1_c, nu, x_O = params
            
            # Check for invalid parameters
            if nu <= 0:
                return 1e10
                
            try:
                # Compute scaled coordinates
                x_scaled = (j2_j1 - j2_j1_c) * np.power(L, 1.0 / nu)
                y_scaled = observable * np.power(L, x_O / nu)
                
                # Compute collapse quality
                chi_sq = self.collapse_quality(x_scaled, y_scaled, L)
                
                return chi_sq
            except (ValueError, RuntimeWarning, FloatingPointError):
                return 1e10
                
        # Set bounds for optimization
        j2_j1_min, j2_j1_max = np.min(j2_j1), np.max(j2_j1)
        j2_j1_c_bounds = (j2_j1_min, j2_j1_max)
        bounds = [j2_j1_c_bounds, nu_bounds, x_O_bounds]
        
        # Optimize using specified method
        if method == 'differential_evolution':
            # Global optimization - more robust but slower
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=1000,
                atol=1e-6,
                tol=1e-6,
                workers=1
            )
        else:
            # Local optimization - faster but may find local minimum
            x0 = [j2_j1_c_init, 1.0, 0.0]  # Initial guess
            result = minimize(
                objective,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
        # Extract results
        j2_j1_c_opt, nu_opt, x_O_opt = result.x
        chi_squared = result.fun
        
        logger.info(f"Scaling collapse optimization: j2_j1_c={j2_j1_c_opt:.4f}, "
                   f"nu={nu_opt:.4f}, x_O={x_O_opt:.4f}, chi^2={chi_squared:.4e}")
        
        return {
            'j2_j1_c': j2_j1_c_opt,
            'nu': nu_opt,
            'x_O': x_O_opt,
            'chi_squared': chi_squared,
            'success': result.success
        }

    def bootstrap_exponents(self,
                           j2_j1: np.ndarray,
                           L: np.ndarray,
                           observable: np.ndarray,
                           j2_j1_c_init: float,
                           n_bootstrap: Optional[int] = None) -> Dict[str, Tuple[float, float]]:
        """
        Bootstrap uncertainty estimation for critical exponents.
        
        Resamples data with replacement and refits scaling parameters
        to estimate uncertainties.
        
        Args:
            j2_j1: Array of frustration ratio values
            L: Array of lattice sizes
            observable: Array of observable values
            j2_j1_c_init: Initial guess for critical point
            n_bootstrap: Number of bootstrap samples (default: from config)
            
        Returns:
            Dictionary with keys:
                - j2_j1_c: (mean, std_error)
                - nu: (mean, std_error)
                - x_O: (mean, std_error)
                - chi_squared: (mean, std_error)
        """
        if n_bootstrap is None:
            n_bootstrap = self.bootstrap_samples
            
        n_points = len(j2_j1)
        
        # Storage for bootstrap results
        j2_j1_c_samples = []
        nu_samples = []
        x_O_samples = []
        chi_squared_samples = []
        
        logger.info(f"Starting bootstrap with {n_bootstrap} samples...")
        
        # Perform bootstrap resampling
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_points, size=n_points, replace=True)
            j2_j1_boot = j2_j1[indices]
            L_boot = L[indices]
            obs_boot = observable[indices]
            
            try:
                # Optimize collapse for this bootstrap sample
                result = self.optimize_collapse(
                    j2_j1_boot,
                    L_boot,
                    obs_boot,
                    j2_j1_c_init,
                    method='minimize'  # Use faster local optimization for bootstrap
                )
                
                if result['success']:
                    j2_j1_c_samples.append(result['j2_j1_c'])
                    nu_samples.append(result['nu'])
                    x_O_samples.append(result['x_O'])
                    chi_squared_samples.append(result['chi_squared'])
                    
            except Exception as e:
                logger.warning(f"Bootstrap sample {i} failed: {e}")
                continue
                
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{n_bootstrap} bootstrap samples")
                
        # Compute statistics
        if len(j2_j1_c_samples) < 10:
            logger.warning(f"Only {len(j2_j1_c_samples)} successful bootstrap samples")
            
        j2_j1_c_mean = np.mean(j2_j1_c_samples)
        j2_j1_c_std = np.std(j2_j1_c_samples, ddof=1)
        
        nu_mean = np.mean(nu_samples)
        nu_std = np.std(nu_samples, ddof=1)
        
        x_O_mean = np.mean(x_O_samples)
        x_O_std = np.std(x_O_samples, ddof=1)
        
        chi_squared_mean = np.mean(chi_squared_samples)
        chi_squared_std = np.std(chi_squared_samples, ddof=1)
        
        logger.info(f"Bootstrap results: j2_j1_c = {j2_j1_c_mean:.4f} ± {j2_j1_c_std:.4f}, "
                   f"nu = {nu_mean:.4f} ± {nu_std:.4f}, "
                   f"x_O = {x_O_mean:.4f} ± {x_O_std:.4f}")
        
        return {
            'j2_j1_c': (j2_j1_c_mean, j2_j1_c_std),
            'nu': (nu_mean, nu_std),
            'x_O': (x_O_mean, x_O_std),
            'chi_squared': (chi_squared_mean, chi_squared_std),
            'n_successful': len(j2_j1_c_samples)
        }

    def extrapolate_to_thermodynamic_limit(self,
                                          j2_j1_c_vs_L: Dict[int, Tuple[float, float]],
                                          nu: float) -> Tuple[float, float]:
        """
        Extrapolate critical point to thermodynamic limit (L → ∞).
        
        Uses the finite-size scaling form:
        j2_j1_c(L) = j2_j1_c(∞) + a * L^(-1/ν)
        
        Fits this form to extract j2_j1_c(∞).
        
        Args:
            j2_j1_c_vs_L: Dictionary mapping L -> (j2_j1_c, uncertainty)
            nu: Correlation length exponent
            
        Returns:
            Tuple of (j2_j1_c_infinity, uncertainty)
        """
        if len(j2_j1_c_vs_L) < 2:
            raise ValueError("Need at least 2 system sizes for extrapolation")
            
        # Extract data
        L_values = np.array(sorted(j2_j1_c_vs_L.keys()))
        j2_j1_c_values = np.array([j2_j1_c_vs_L[L][0] for L in L_values])
        uncertainties = np.array([j2_j1_c_vs_L[L][1] for L in L_values])
        
        # Compute x = L^(-1/ν)
        x = np.power(L_values, -1.0 / nu)
        
        # Weighted linear fit: j2_j1_c(L) = j2_j1_c_inf + a * x
        # Use inverse variance weighting
        weights = 1.0 / (uncertainties**2 + 1e-10)  # Add small constant to avoid division by zero
        
        # Compute weighted means
        w_sum = np.sum(weights)
        x_mean = np.sum(weights * x) / w_sum
        y_mean = np.sum(weights * j2_j1_c_values) / w_sum
        
        # Compute slope and intercept
        numerator = np.sum(weights * (x - x_mean) * (j2_j1_c_values - y_mean))
        denominator = np.sum(weights * (x - x_mean)**2)
        
        if np.abs(denominator) < 1e-10:
            # Degenerate case - all points have same x value
            j2_j1_c_inf = y_mean
            uncertainty_inf = np.sqrt(np.sum((uncertainties * weights)**2)) / w_sum
        else:
            slope = numerator / denominator
            j2_j1_c_inf = y_mean - slope * x_mean  # Intercept at x=0 (L=∞)
            
            # Estimate uncertainty from fit residuals
            predictions = j2_j1_c_inf + slope * x
            residuals = j2_j1_c_values - predictions
            chi_squared = np.sum(weights * residuals**2)
            
            # Uncertainty in intercept
            variance_inf = chi_squared / (len(L_values) - 2) * (1.0/w_sum + x_mean**2/denominator)
            uncertainty_inf = np.sqrt(variance_inf)
            
        logger.info(f"Extrapolated critical point: j2_j1_c(∞) = {j2_j1_c_inf:.4f} ± {uncertainty_inf:.4f}")
        
        return j2_j1_c_inf, uncertainty_inf
    
    def get_scaled_data(self,
                       j2_j1: np.ndarray,
                       L: np.ndarray,
                       observable: np.ndarray,
                       j2_j1_c: float,
                       nu: float,
                       x_O: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data to scaled coordinates for plotting.
        
        Args:
            j2_j1: Array of frustration ratio values
            L: Array of lattice sizes
            observable: Array of observable values
            j2_j1_c: Critical point
            nu: Correlation length exponent
            x_O: Scaling dimension
            
        Returns:
            Tuple of (x_scaled, y_scaled) arrays
        """
        x_scaled = (j2_j1 - j2_j1_c) * np.power(L, 1.0 / nu)
        y_scaled = observable * np.power(L, x_O / nu)
        
        return x_scaled, y_scaled
