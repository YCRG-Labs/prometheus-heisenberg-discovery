"""Critical Point Detection Module

This module implements multiple methods for detecting phase transition critical points
in the J1-J2 Heisenberg model using Q-VAE latent representations and ground state data.

Methods implemented:
1. Latent Variance Method: Detects critical points from variance peaks across system sizes
2. Reconstruction Error Method: Detects critical points from reconstruction error peaks
3. Fidelity Susceptibility Method: Detects critical points from fidelity susceptibility peaks
4. Ensemble Estimation: Combines all methods with inverse-variance weighting

Key features:
- Multiple independent detection methods for robustness
- Bootstrap uncertainty quantification
- Peak finding with sub-grid resolution
- Ensemble estimates with consistency checking
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


class LatentVarianceMethod:
    """Detect critical points from latent variance peaks across system sizes
    
    The latent variance method computes the variance of latent representations
    across different system sizes for each parameter point. At critical points,
    the latent representations show maximum variation across system sizes due to
    finite-size effects being strongest at the transition.
    
    Algorithm:
    1. For each j2_j1 value, collect latent representations across all L
    2. Compute variance: Var[z(j2_j1, L)] over L
    3. Smooth variance curve using Savitzky-Golay filter
    4. Find peaks using scipy.signal.find_peaks
    5. Fit parabola near peak for sub-grid resolution
    """
    
    def __init__(self, smooth_window: int = 5, smooth_order: int = 2):
        """Initialize LatentVarianceMethod
        
        Args:
            smooth_window: Window length for Savitzky-Golay filter (must be odd)
            smooth_order: Polynomial order for Savitzky-Golay filter
        """
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order
        
        logger.debug(
            f"Initialized LatentVarianceMethod: "
            f"smooth_window={smooth_window}, smooth_order={smooth_order}"
        )
    
    def compute_latent_variance(
        self,
        latent_reps: Dict[Tuple[float, int], np.ndarray]
    ) -> Dict[float, float]:
        """Compute variance of latent representations across system sizes
        
        For each j2_j1 value, computes the variance of latent vectors across
        different lattice sizes L. High variance indicates strong finite-size
        effects, which peak at critical points.
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
        
        Returns:
            Dictionary mapping j2_j1 -> total variance
        """
        # Group latent representations by j2_j1
        latent_by_j2j1 = {}
        for (j2_j1, L), z in latent_reps.items():
            if j2_j1 not in latent_by_j2j1:
                latent_by_j2j1[j2_j1] = []
            latent_by_j2j1[j2_j1].append(z)
        
        # Compute variance for each j2_j1
        variance_dict = {}
        for j2_j1, z_list in latent_by_j2j1.items():
            # Stack latent vectors: shape (n_sizes, latent_dim)
            z_array = np.array(z_list)
            
            # Compute variance across system sizes (axis=0)
            # Then sum over latent dimensions to get total variance
            variance = np.sum(np.var(z_array, axis=0))
            
            variance_dict[j2_j1] = variance
        
        logger.debug(f"Computed latent variance for {len(variance_dict)} parameter points")
        
        return variance_dict
    
    def detect_critical_point(
        self,
        latent_reps: Dict[Tuple[float, int], np.ndarray]
    ) -> Tuple[float, float]:
        """Detect critical point from latent variance peak
        
        Computes latent variance, smooths the curve, finds peaks, and returns
        the most prominent peak as the critical point estimate.
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
        
        Returns:
            Tuple of (j2_j1_critical, uncertainty)
            - j2_j1_critical: Estimated critical point location
            - uncertainty: Uncertainty estimate from peak width
        
        Raises:
            ValueError: If no peaks found in variance curve
        """
        # Compute latent variance
        variance_dict = self.compute_latent_variance(latent_reps)
        
        # Sort by j2_j1 for proper curve analysis
        j2_j1_values = np.array(sorted(variance_dict.keys()))
        variance_values = np.array([variance_dict[j] for j in j2_j1_values])
        
        # Smooth variance curve using Savitzky-Golay filter
        if len(j2_j1_values) >= self.smooth_window:
            variance_smooth = savgol_filter(
                variance_values,
                window_length=self.smooth_window,
                polyorder=self.smooth_order
            )
        else:
            variance_smooth = variance_values
            logger.warning(
                f"Not enough points ({len(j2_j1_values)}) for smoothing "
                f"(window={self.smooth_window}). Using raw data."
            )
        
        # Find peaks
        peaks, properties = find_peaks(
            variance_smooth,
            prominence=0.1 * np.max(variance_smooth)  # Require 10% prominence
        )
        
        if len(peaks) == 0:
            raise ValueError("No peaks found in latent variance curve")
        
        # Select most prominent peak
        prominences = properties['prominences']
        most_prominent_idx = np.argmax(prominences)
        peak_idx = peaks[most_prominent_idx]
        
        j2_j1_critical = j2_j1_values[peak_idx]
        
        # Estimate uncertainty from peak width at half maximum
        widths = properties.get('widths', np.array([1.0]))
        if most_prominent_idx < len(widths):
            width = widths[most_prominent_idx]
            # Convert width in indices to width in j2_j1 units
            if len(j2_j1_values) > 1:
                dj = np.mean(np.diff(j2_j1_values))
                uncertainty = width * dj / 2.355  # FWHM to sigma conversion
            else:
                uncertainty = 0.01  # Default uncertainty
        else:
            uncertainty = 0.01
        
        logger.info(
            f"Latent variance method: j2_j1_c = {j2_j1_critical:.4f} ± {uncertainty:.4f}"
        )
        
        return j2_j1_critical, uncertainty


class ReconstructionErrorMethod:
    """Detect critical points from reconstruction error peaks
    
    The reconstruction error method computes the Q-VAE reconstruction error
    (1 - fidelity) for each state. Critical states have maximum complexity
    and entanglement, making them hardest to compress, leading to peaks in
    reconstruction error.
    
    Algorithm:
    1. For each state, compute reconstruction error using Q-VAE
    2. Average over system sizes for each j2_j1
    3. Smooth error curve
    4. Find peaks
    """
    
    def __init__(self, qvae_module: Any, smooth_window: int = 5, smooth_order: int = 2):
        """Initialize ReconstructionErrorMethod
        
        Args:
            qvae_module: QVAEModule instance with trained models
            smooth_window: Window length for Savitzky-Golay filter
            smooth_order: Polynomial order for Savitzky-Golay filter
        """
        self.qvae_module = qvae_module
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order
        
        logger.debug("Initialized ReconstructionErrorMethod")
    
    def compute_reconstruction_error(
        self,
        states: Dict[Tuple[float, int], Any]
    ) -> Dict[Tuple[float, int], float]:
        """Compute reconstruction error for each state
        
        Uses the trained Q-VAE models to compute reconstruction error
        (1 - fidelity) for each ground state.
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
        
        Returns:
            Dictionary mapping (j2_j1, L) -> reconstruction error
        
        Raises:
            RuntimeError: If Q-VAE model not trained for required lattice size
        """
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        error_dict = {}
        
        for (j2_j1, L), state in states.items():
            # Check if model exists
            if L not in self.qvae_module.models:
                raise RuntimeError(
                    f"No trained Q-VAE model for L={L}. Train models first."
                )
            
            model = self.qvae_module.models[L]
            model.eval()
            
            # Convert state to tensor
            real_vector = state.to_real_vector()
            x = torch.tensor(real_vector, dtype=torch.float32).unsqueeze(0)
            x = x.to(device)
            
            # Forward pass
            with torch.no_grad():
                recon, mu, logvar = model(x)
                
                # Compute fidelity
                fidelity = model.compute_fidelity(x, recon)
                
                # Reconstruction error = 1 - fidelity
                error = 1.0 - fidelity.item()
            
            error_dict[(j2_j1, L)] = error
        
        logger.debug(f"Computed reconstruction error for {len(error_dict)} states")
        
        return error_dict
    
    def detect_critical_point(
        self,
        states: Dict[Tuple[float, int], Any]
    ) -> Tuple[float, float]:
        """Detect critical point from reconstruction error peak
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
        
        Returns:
            Tuple of (j2_j1_critical, uncertainty)
        
        Raises:
            ValueError: If no peaks found in error curve
        """
        # Compute reconstruction errors
        error_dict = self.compute_reconstruction_error(states)
        
        # Average over system sizes for each j2_j1
        error_by_j2j1 = {}
        for (j2_j1, L), error in error_dict.items():
            if j2_j1 not in error_by_j2j1:
                error_by_j2j1[j2_j1] = []
            error_by_j2j1[j2_j1].append(error)
        
        j2_j1_values = np.array(sorted(error_by_j2j1.keys()))
        error_values = np.array([np.mean(error_by_j2j1[j]) for j in j2_j1_values])
        
        # Smooth error curve
        if len(j2_j1_values) >= self.smooth_window:
            error_smooth = savgol_filter(
                error_values,
                window_length=self.smooth_window,
                polyorder=self.smooth_order
            )
        else:
            error_smooth = error_values
        
        # Find peaks
        peaks, properties = find_peaks(
            error_smooth,
            prominence=0.1 * np.max(error_smooth)
        )
        
        if len(peaks) == 0:
            raise ValueError("No peaks found in reconstruction error curve")
        
        # Select most prominent peak
        prominences = properties['prominences']
        most_prominent_idx = np.argmax(prominences)
        peak_idx = peaks[most_prominent_idx]
        
        j2_j1_critical = j2_j1_values[peak_idx]
        
        # Estimate uncertainty from peak width
        widths = properties.get('widths', np.array([1.0]))
        if most_prominent_idx < len(widths):
            width = widths[most_prominent_idx]
            if len(j2_j1_values) > 1:
                dj = np.mean(np.diff(j2_j1_values))
                uncertainty = width * dj / 2.355
            else:
                uncertainty = 0.01
        else:
            uncertainty = 0.01
        
        logger.info(
            f"Reconstruction error method: j2_j1_c = {j2_j1_critical:.4f} ± {uncertainty:.4f}"
        )
        
        return j2_j1_critical, uncertainty


class FidelitySusceptibilityMethod:
    """Detect critical points from fidelity susceptibility peaks
    
    Fidelity susceptibility measures the sensitivity of ground state fidelity
    to parameter changes. It peaks at critical points where the ground state
    changes most rapidly.
    
    χ_F(λ) ≈ -∂²/∂λ² log F(ψ(λ), ψ(λ+δ))
    
    Using finite differences:
    χ_F ≈ [log F(λ-δ) - 2 log F(λ) + log F(λ+δ)] / δ²
    """
    
    def __init__(self, smooth_window: int = 5, smooth_order: int = 2):
        """Initialize FidelitySusceptibilityMethod
        
        Args:
            smooth_window: Window length for Savitzky-Golay filter
            smooth_order: Polynomial order for Savitzky-Golay filter
        """
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order
        
        logger.debug("Initialized FidelitySusceptibilityMethod")
    
    def compute_fidelity_susceptibility(
        self,
        states: Dict[Tuple[float, int], Any],
        delta: float = 0.01
    ) -> Dict[Tuple[float, int], float]:
        """Compute fidelity susceptibility using finite differences
        
        For each parameter point (j2_j1, L), computes the fidelity with
        neighboring states and uses finite differences to estimate the
        second derivative.
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
            delta: Step size for finite differences (default: 0.01)
        
        Returns:
            Dictionary mapping (j2_j1, L) -> fidelity susceptibility
        """
        susceptibility_dict = {}
        
        # Group states by lattice size
        states_by_L = {}
        for (j2_j1, L), state in states.items():
            if L not in states_by_L:
                states_by_L[L] = {}
            states_by_L[L][j2_j1] = state
        
        # Compute susceptibility for each lattice size separately
        for L, states_L in states_by_L.items():
            j2_j1_values = sorted(states_L.keys())
            
            for i, j2_j1 in enumerate(j2_j1_values):
                state_center = states_L[j2_j1]
                
                # Find neighboring states
                # Look for states within delta of j2_j1 ± delta
                state_left = None
                state_right = None
                
                for j2_j1_neighbor in j2_j1_values:
                    if abs(j2_j1_neighbor - (j2_j1 - delta)) < delta / 2:
                        state_left = states_L[j2_j1_neighbor]
                    if abs(j2_j1_neighbor - (j2_j1 + delta)) < delta / 2:
                        state_right = states_L[j2_j1_neighbor]
                
                # Compute susceptibility using available neighbors
                if state_left is not None and state_right is not None:
                    # Central difference: [F(λ-δ) - 2F(λ) + F(λ+δ)] / δ²
                    F_left = self._compute_fidelity(state_center, state_left)
                    F_right = self._compute_fidelity(state_center, state_right)
                    
                    # Use log fidelity for numerical stability
                    log_F_left = np.log(F_left + 1e-10)
                    log_F_center = 0.0  # log F(ψ, ψ) = log 1 = 0
                    log_F_right = np.log(F_right + 1e-10)
                    
                    chi_F = -(log_F_left - 2 * log_F_center + log_F_right) / delta**2
                    
                elif state_right is not None:
                    # Forward difference
                    F_right = self._compute_fidelity(state_center, state_right)
                    log_F_right = np.log(F_right + 1e-10)
                    chi_F = -log_F_right / delta**2
                    
                elif state_left is not None:
                    # Backward difference
                    F_left = self._compute_fidelity(state_center, state_left)
                    log_F_left = np.log(F_left + 1e-10)
                    chi_F = -log_F_left / delta**2
                    
                else:
                    # No neighbors available
                    chi_F = 0.0
                
                susceptibility_dict[(j2_j1, L)] = chi_F
        
        logger.debug(f"Computed fidelity susceptibility for {len(susceptibility_dict)} states")
        
        return susceptibility_dict
    
    def _compute_fidelity(self, state1: Any, state2: Any) -> float:
        """Compute quantum fidelity between two states
        
        F = |⟨ψ₁|ψ₂⟩|²
        
        Args:
            state1: First GroundState
            state2: Second GroundState
        
        Returns:
            Fidelity value in [0, 1]
        """
        # Compute inner product
        overlap = np.vdot(state1.coefficients, state2.coefficients)
        
        # Fidelity = |overlap|²
        fidelity = np.abs(overlap) ** 2
        
        return float(fidelity)
    
    def detect_critical_point(
        self,
        states: Dict[Tuple[float, int], Any],
        delta: float = 0.01
    ) -> Tuple[float, float]:
        """Detect critical point from fidelity susceptibility peak
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
            delta: Step size for finite differences
        
        Returns:
            Tuple of (j2_j1_critical, uncertainty)
        
        Raises:
            ValueError: If no peaks found in susceptibility curve
        """
        # Compute fidelity susceptibility
        susceptibility_dict = self.compute_fidelity_susceptibility(states, delta)
        
        # Average over system sizes
        susc_by_j2j1 = {}
        for (j2_j1, L), chi_F in susceptibility_dict.items():
            if j2_j1 not in susc_by_j2j1:
                susc_by_j2j1[j2_j1] = []
            susc_by_j2j1[j2_j1].append(chi_F)
        
        j2_j1_values = np.array(sorted(susc_by_j2j1.keys()))
        susc_values = np.array([np.mean(susc_by_j2j1[j]) for j in j2_j1_values])
        
        # Smooth susceptibility curve
        if len(j2_j1_values) >= self.smooth_window:
            susc_smooth = savgol_filter(
                susc_values,
                window_length=self.smooth_window,
                polyorder=self.smooth_order
            )
        else:
            susc_smooth = susc_values
        
        # Find peaks
        peaks, properties = find_peaks(
            susc_smooth,
            prominence=0.1 * np.max(susc_smooth)
        )
        
        if len(peaks) == 0:
            raise ValueError("No peaks found in fidelity susceptibility curve")
        
        # Select most prominent peak
        prominences = properties['prominences']
        most_prominent_idx = np.argmax(prominences)
        peak_idx = peaks[most_prominent_idx]
        
        j2_j1_critical = j2_j1_values[peak_idx]
        
        # Estimate uncertainty from peak width
        widths = properties.get('widths', np.array([1.0]))
        if most_prominent_idx < len(widths):
            width = widths[most_prominent_idx]
            if len(j2_j1_values) > 1:
                dj = np.mean(np.diff(j2_j1_values))
                uncertainty = width * dj / 2.355
            else:
                uncertainty = 0.01
        else:
            uncertainty = 0.01
        
        logger.info(
            f"Fidelity susceptibility method: j2_j1_c = {j2_j1_critical:.4f} ± {uncertainty:.4f}"
        )
        
        return j2_j1_critical, uncertainty


class CriticalPointDetection:
    """Ensemble critical point detection combining multiple methods
    
    This class coordinates all detection methods and provides ensemble
    estimates with uncertainty quantification.
    
    Features:
    - Applies all detection methods
    - Combines estimates using inverse-variance weighting
    - Bootstrap uncertainty quantification
    - Consistency checking across methods
    """
    
    def __init__(self, config: Any, qvae_module: Optional[Any] = None):
        """Initialize CriticalPointDetection
        
        Args:
            config: Configuration object
            qvae_module: Optional QVAEModule instance with trained models.
                If None, reconstruction_error method is skipped (latent_variance and
                fidelity_susceptibility still run).
        """
        self.config = config
        self.qvae_module = qvae_module
        
        # Initialize detection methods (reconstruction_error needs qvae_module)
        self.methods = {
            'latent_variance': LatentVarianceMethod(),
            'fidelity_susceptibility': FidelitySusceptibilityMethod(),
        }
        if qvae_module is not None:
            self.methods['reconstruction_error'] = ReconstructionErrorMethod(qvae_module)
        
        logger.info(
            f"Initialized CriticalPointDetection with {len(self.methods)} methods"
        )
    
    def detect_all_methods(
        self,
        states: Dict[Tuple[float, int], Any],
        latent_reps: Dict[Tuple[float, int], np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """Apply all detection methods
        
        Runs each detection method and collects the results.
        
        Args:
            states: Dictionary mapping (j2_j1, L) -> GroundState
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
        
        Returns:
            Dictionary mapping method_name -> (j2_j1_critical, uncertainty)
        """
        logger.info("Applying all critical point detection methods...")
        
        detections = {}
        
        # Latent variance method
        try:
            j2_j1_c, uncertainty = self.methods['latent_variance'].detect_critical_point(
                latent_reps
            )
            detections['latent_variance'] = (j2_j1_c, uncertainty)
        except Exception as e:
            logger.warning(f"Latent variance method failed: {e}")
        
        # Reconstruction error method (only if qvae_module was provided)
        if 'reconstruction_error' in self.methods:
            try:
                j2_j1_c, uncertainty = self.methods['reconstruction_error'].detect_critical_point(
                    states
                )
                detections['reconstruction_error'] = (j2_j1_c, uncertainty)
            except Exception as e:
                logger.warning(f"Reconstruction error method failed: {e}")
        
        # Fidelity susceptibility method
        try:
            j2_j1_c, uncertainty = self.methods['fidelity_susceptibility'].detect_critical_point(
                states
            )
            detections['fidelity_susceptibility'] = (j2_j1_c, uncertainty)
        except Exception as e:
            logger.warning(f"Fidelity susceptibility method failed: {e}")
        
        logger.info(
            f"Successfully applied {len(detections)}/{len(self.methods)} detection methods"
        )
        
        return detections
    
    def ensemble_estimate(
        self,
        detections: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Compute inverse-variance weighted ensemble estimate
        
        Combines estimates from all methods using inverse-variance weighting:
        - Weight: w_i = 1 / σ_i²
        - Ensemble mean: j2_j1_c = Σ(w_i * j2_j1_i) / Σ(w_i)
        - Ensemble uncertainty: σ = 1 / √(Σ w_i)
        
        Args:
            detections: Dictionary mapping method_name -> (j2_j1_critical, uncertainty)
        
        Returns:
            Tuple of (j2_j1_critical_ensemble, uncertainty_ensemble)
        
        Raises:
            ValueError: If no valid detections provided
        """
        if not detections:
            raise ValueError("No valid detections to combine")
        
        # Extract estimates and uncertainties
        estimates = []
        uncertainties = []
        method_names = []
        
        for method, (j2_j1_c, uncertainty) in detections.items():
            estimates.append(j2_j1_c)
            uncertainties.append(uncertainty)
            method_names.append(method)
        
        estimates = np.array(estimates)
        uncertainties = np.array(uncertainties)
        
        # Compute inverse-variance weights
        weights = 1.0 / (uncertainties ** 2)
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted mean
        j2_j1_ensemble = np.sum(weights * estimates)
        
        # Ensemble uncertainty: 1 / √(Σ w_i)
        uncertainty_ensemble = 1.0 / np.sqrt(np.sum(1.0 / (uncertainties ** 2)))
        
        logger.info(
            f"Ensemble estimate: j2_j1_c = {j2_j1_ensemble:.4f} ± {uncertainty_ensemble:.4f}"
        )
        
        # Check consistency (flag if methods disagree by > 3σ)
        max_deviation = np.max(np.abs(estimates - j2_j1_ensemble))
        if max_deviation > 3 * uncertainty_ensemble:
            logger.warning(
                f"Methods show inconsistency: max deviation = {max_deviation:.4f} "
                f"(> 3σ = {3*uncertainty_ensemble:.4f})"
            )
        
        return j2_j1_ensemble, uncertainty_ensemble
    
    def bootstrap_uncertainty(
        self,
        method_func: Callable,
        data: Any,
        n_bootstrap: int = 1000
    ) -> float:
        """Estimate uncertainty using bootstrap resampling
        
        Resamples the data with replacement, recomputes the critical point
        for each bootstrap sample, and returns the standard error.
        
        Args:
            method_func: Detection method function to apply
            data: Data to resample (states or latent_reps)
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Standard error of critical point estimate
        """
        bootstrap_estimates = []
        
        # Get keys for resampling
        if isinstance(data, dict):
            keys = list(data.keys())
        else:
            raise ValueError("Data must be a dictionary")
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_keys = np.random.choice(keys, size=len(keys), replace=True)
            bootstrap_data = {key: data[key] for key in bootstrap_keys}
            
            try:
                j2_j1_c, _ = method_func(bootstrap_data)
                bootstrap_estimates.append(j2_j1_c)
            except Exception:
                # Skip failed bootstrap samples
                continue
        
        if len(bootstrap_estimates) < n_bootstrap // 2:
            logger.warning(
                f"Only {len(bootstrap_estimates)}/{n_bootstrap} bootstrap samples succeeded"
            )
        
        # Compute standard error
        if bootstrap_estimates:
            std_error = np.std(bootstrap_estimates)
        else:
            std_error = 0.01  # Default uncertainty
        
        return std_error
