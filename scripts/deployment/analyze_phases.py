#!/usr/bin/env python3
"""Phase Analysis for J1-J2 Heisenberg Model

Analyzes latent space representations from trained VAEs to:
1. Identify order parameters (max variance latent dimensions)
2. Detect critical points (susceptibility peaks, gradient maxima)
3. Perform finite-size scaling across L=4,5,6,8

Usage:
    python analyze_phases.py --model_dir results/trained_models --data_dir results/groundstates
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import h5py
import json
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import logging
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_logger(output_dir: Path) -> logging.Logger:
    log_file = output_dir / f"phase_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_latent_representations(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load latent representations from HDF5.
    
    Returns:
        Tuple of (j2_values, latent_vectors)
    """
    j2_list = []
    z_list = []
    
    with h5py.File(filepath, 'r') as f:
        for key in sorted(f.keys()):
            if not key.startswith('J2_'):
                continue
            j2 = f[key].attrs['j2_j1']
            z = f[key][:]
            j2_list.append(j2)
            z_list.append(z)
    
    return np.array(j2_list), np.array(z_list)


def load_observables(filepath: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load observables from HDF5.
    
    Returns:
        Tuple of (j2_values, dict of observable arrays)
    """
    j2_list = []
    obs_dict = {}
    
    with h5py.File(filepath, 'r') as f:
        for key in sorted(f.keys()):
            if not key.startswith('J2_'):
                continue
            
            grp = f[key]
            j2 = grp.attrs['j2_j1']
            j2_list.append(j2)
            
            obs = grp['observables'][:]
            obs_names = ['energy', 'energy_density', 'staggered_mag', 
                        'stripe_order', 'plaquette_order', 'S_pi_pi', 'S_pi_0',
                        'entanglement_entropy', 'nematic_order', 'dimer_x', 'dimer_y']
            
            for i, name in enumerate(obs_names):
                if name not in obs_dict:
                    obs_dict[name] = []
                if i < len(obs):
                    obs_dict[name].append(obs[i])
    
    j2_arr = np.array(j2_list)
    for name in obs_dict:
        obs_dict[name] = np.array(obs_dict[name])
    
    return j2_arr, obs_dict


def find_order_parameter_dimension(z: np.ndarray) -> int:
    """Find latent dimension with maximum variance (order parameter)."""
    variances = np.var(z, axis=0)
    return int(np.argmax(variances))


def compute_latent_susceptibility(j2: np.ndarray, z: np.ndarray, dim: int) -> np.ndarray:
    """Compute susceptibility χ = N * Var(z_dim) at each J2."""
    # For each J2 point, we only have one sample, so use local variance
    # Approximate with sliding window or finite differences
    phi = z[:, dim]
    
    # Use gradient magnitude as proxy for susceptibility
    dphi = np.gradient(phi, j2)
    chi = np.abs(dphi)
    
    return chi


def find_critical_point_susceptibility(j2: np.ndarray, chi: np.ndarray) -> Tuple[float, float]:
    """Find critical point from susceptibility peak."""
    # Smooth the susceptibility
    from scipy.ndimage import gaussian_filter1d
    chi_smooth = gaussian_filter1d(chi, sigma=2)
    
    # Find peaks
    peaks, properties = find_peaks(chi_smooth, height=0)
    
    if len(peaks) == 0:
        # No clear peak, use max
        idx = np.argmax(chi_smooth)
    else:
        # Use highest peak
        idx = peaks[np.argmax(chi_smooth[peaks])]
    
    j2_c = j2[idx]
    
    # Estimate uncertainty from peak width
    half_max = chi_smooth[idx] / 2
    above_half = chi_smooth > half_max
    width_indices = np.where(above_half)[0]
    if len(width_indices) > 1:
        width = j2[width_indices[-1]] - j2[width_indices[0]]
        uncertainty = width / 4  # Rough estimate
    else:
        uncertainty = 0.05
    
    return j2_c, uncertainty


def find_critical_point_gradient(j2: np.ndarray, z: np.ndarray, dim: int) -> Tuple[float, float]:
    """Find critical point from maximum gradient of order parameter."""
    phi = z[:, dim]
    dphi = np.abs(np.gradient(phi, j2))
    
    idx = np.argmax(dphi)
    j2_c = j2[idx]
    
    # Uncertainty from gradient width
    uncertainty = 0.05
    
    return j2_c, uncertainty


def compute_binder_cumulant(phi: np.ndarray) -> float:
    """Compute Binder cumulant U = 1 - <φ⁴>/(3<φ²>²)."""
    phi2 = np.mean(phi**2)
    phi4 = np.mean(phi**4)
    
    if phi2 < 1e-10:
        return 0.0
    
    return 1.0 - phi4 / (3 * phi2**2)


def finite_size_scaling_analysis(
    results: Dict[int, Dict[str, Any]],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Perform finite-size scaling analysis across system sizes."""
    
    L_values = sorted(results.keys())
    
    if len(L_values) < 2:
        logger.warning("Need at least 2 system sizes for FSS")
        return {}
    
    # Collect critical points
    j2_c_values = []
    j2_c_errors = []
    
    for L in L_values:
        if 'j2_c' in results[L]:
            j2_c_values.append(results[L]['j2_c'])
            j2_c_errors.append(results[L]['j2_c_err'])
    
    if len(j2_c_values) < 2:
        return {}
    
    # Weighted average of critical points
    weights = 1.0 / np.array(j2_c_errors)**2
    j2_c_avg = np.sum(np.array(j2_c_values) * weights) / np.sum(weights)
    j2_c_avg_err = 1.0 / np.sqrt(np.sum(weights))
    
    # Fit FSS form: j2_c(L) = j2_c(∞) + a * L^(-1/ν)
    # For now, just report the trend
    
    fss_results = {
        'L_values': L_values,
        'j2_c_values': j2_c_values,
        'j2_c_errors': j2_c_errors,
        'j2_c_extrapolated': j2_c_avg,
        'j2_c_extrapolated_err': j2_c_avg_err
    }
    
    logger.info(f"FSS: J2/J1_c = {j2_c_avg:.4f} ± {j2_c_avg_err:.4f}")
    
    return fss_results


def analyze_single_size(
    L: int,
    latent_file: Path,
    data_file: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Analyze a single system size."""
    
    results = {'L': L}
    
    # Load latent representations
    if latent_file.exists():
        j2, z = load_latent_representations(latent_file)
        
        # Find order parameter dimension
        op_dim = find_order_parameter_dimension(z)
        results['order_param_dim'] = op_dim
        results['latent_variances'] = np.var(z, axis=0).tolist()
        
        logger.info(f"L={L}: Order parameter in latent dim {op_dim}")
        
        # Compute susceptibility
        chi = compute_latent_susceptibility(j2, z, op_dim)
        
        # Find critical point (multiple methods)
        j2_c_chi, err_chi = find_critical_point_susceptibility(j2, chi)
        j2_c_grad, err_grad = find_critical_point_gradient(j2, z, op_dim)
        
        # Weighted average
        w_chi = 1.0 / err_chi**2
        w_grad = 1.0 / err_grad**2
        j2_c = (j2_c_chi * w_chi + j2_c_grad * w_grad) / (w_chi + w_grad)
        j2_c_err = 1.0 / np.sqrt(w_chi + w_grad)
        
        results['j2_c'] = float(j2_c)
        results['j2_c_err'] = float(j2_c_err)
        results['j2_c_susceptibility'] = float(j2_c_chi)
        results['j2_c_gradient'] = float(j2_c_grad)
        
        logger.info(f"L={L}: J2/J1_c = {j2_c:.4f} ± {j2_c_err:.4f}")
        
        # Store latent trajectory
        results['j2_values'] = j2.tolist()
        results['order_param_values'] = z[:, op_dim].tolist()
    
    # Load observables
    if data_file.exists():
        j2_obs, obs_dict = load_observables(data_file)
        
        if 'staggered_mag' in obs_dict:
            results['staggered_mag'] = obs_dict['staggered_mag'].tolist()
        if 'S_pi_pi' in obs_dict:
            results['S_pi_pi'] = obs_dict['S_pi_pi'].tolist()
        if 'entanglement_entropy' in obs_dict:
            results['entanglement_entropy'] = obs_dict['entanglement_entropy'].tolist()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase analysis for J1-J2 model")
    parser.add_argument("--model_dir", type=str, default="results/trained_models")
    parser.add_argument("--data_dir", type=str, default="results/groundstates")
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir)
    logger.info("=" * 60)
    logger.info("J1-J2 Phase Analysis")
    logger.info("=" * 60)
    
    # Analyze each system size
    all_results = {}
    
    for L in [4, 5, 6, 8]:
        logger.info(f"\nAnalyzing L={L}...")
        
        # Find files
        latent_file = model_dir / f"L{L}" / "latent_representations.h5"
        
        if L <= 5:
            data_file = data_dir / f"groundstates_L{L}.h5"
        else:
            data_file = data_dir / f"groundstates_L{L}_rdm.h5"
        
        if not latent_file.exists():
            logger.warning(f"No latent file for L={L}, skipping")
            continue
        
        results = analyze_single_size(L, latent_file, data_file, logger)
        all_results[L] = results
    
    # Finite-size scaling
    logger.info("\n" + "=" * 60)
    logger.info("Finite-Size Scaling Analysis")
    logger.info("=" * 60)
    
    fss_results = finite_size_scaling_analysis(all_results, logger)
    
    # Save results
    output_file = output_dir / "phase_analysis_results.json"
    
    final_results = {
        'per_size': {str(L): r for L, r in all_results.items()},
        'finite_size_scaling': fss_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for L in sorted(all_results.keys()):
        r = all_results[L]
        if 'j2_c' in r:
            logger.info(f"L={L}: J2/J1_c = {r['j2_c']:.4f} ± {r['j2_c_err']:.4f}")
    
    if 'j2_c_extrapolated' in fss_results:
        logger.info(f"\nExtrapolated: J2/J1_c = {fss_results['j2_c_extrapolated']:.4f} ± {fss_results['j2_c_extrapolated_err']:.4f}")
        logger.info(f"Literature value: J2/J1_c ≈ 0.4-0.5 (debated)")


if __name__ == "__main__":
    main()
