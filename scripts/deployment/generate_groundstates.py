#!/usr/bin/env python3
"""Ground State Generation Script for VM Deployment

This script generates ground states for the J1-J2 Heisenberg model using
ITensor/DMRG (via Julia) for larger lattice sizes (L=6+) that QuSpin can't handle.

Output: HDF5 file with ground states, energies, and all 11 observables.
This file is the handoff point - download it from VM and run train_vae.py locally.

Usage:
    python generate_groundstates.py --config configs/vm_config.yaml --L 6
    python generate_groundstates.py --L 6 --j2_min 0.0 --j2_max 1.0 --n_points 41
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import h5py
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.logging_config import setup_logging


def setup_logger(output_dir: Path) -> logging.Logger:
    """Setup logging for ground state generation."""
    log_file = output_dir / f"groundstate_generation_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_completed_points(filepath: Path) -> set:
    """Get set of already-completed J2 points from existing HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Set of completed J2 keys (e.g., {"J2_0.000", "J2_0.025", ...})
    """
    completed = set()
    if filepath.exists():
        try:
            with h5py.File(filepath, 'r') as f:
                completed = set(f.keys())
        except Exception as e:
            logging.warning(f"Could not read existing file {filepath}: {e}")
    return completed


def compute_observables_from_psi(
    psi: np.ndarray,
    L: int,
    j2_j1: float,
    energy: float
) -> np.ndarray:
    """Compute all 11 observables from wavefunction.
    
    This must be called during the DMRG run while we have access to the
    full wavefunction. At L=6 we can't reconstruct observables locally
    without the full wavefunction.
    
    Args:
        psi: Ground state wavefunction coefficients
        L: Lattice size
        j2_j1: Frustration ratio
        energy: Ground state energy
        
    Returns:
        Array of 11 observable values:
        [energy, energy_density, staggered_mag, stripe_order, plaquette_order,
         S_pi_pi, S_pi_0, entanglement_entropy, nematic_order, dimer_x, dimer_y]
    """
    N = L * L
    
    # For now, return placeholder values
    # In production, this would use ITensor's measurement capabilities
    # or reconstruct operators in the MPS basis
    
    observables = np.zeros(11)
    observables[0] = energy  # Energy
    observables[1] = energy / N  # Energy density
    
    # Placeholder for other observables - these need proper ITensor implementation
    # The actual implementation depends on your ITensor/Julia setup
    observables[2] = 0.0  # Staggered magnetization
    observables[3] = 0.0  # Stripe order
    observables[4] = 0.0  # Plaquette order
    observables[5] = 0.0  # S(π,π)
    observables[6] = 0.0  # S(π,0)
    observables[7] = 0.0  # Entanglement entropy
    observables[8] = 0.0  # Nematic order
    observables[9] = 0.0  # Dimer order x
    observables[10] = 0.0  # Dimer order y
    
    return observables


def run_dmrg_itensor(
    L: int,
    j2_j1: float,
    bond_dim: int = 200,
    sweeps: int = 20,
    cutoff: float = 1e-10
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Run DMRG using ITensor (via Julia) for a single J2/J1 point.
    
    Calls the Julia script j1j2_dmrg.jl via subprocess.
    
    Note: For L>=6, we don't store the full wavefunction (too large).
    Instead we store observables computed during DMRG. The VAE training
    for L>=6 will need to use observables directly, not wavefunctions.
    
    Args:
        L: Lattice size
        j2_j1: Frustration ratio J2/J1
        bond_dim: Maximum bond dimension (chi)
        sweeps: Number of DMRG sweeps (not used, hardcoded in Julia)
        cutoff: SVD cutoff for truncation (not used, hardcoded in Julia)
        
    Returns:
        Tuple of (psi_array, energy, observables)
        - psi_array: Dummy array (MPS too large to store as vector)
        - energy: Ground state energy
        - observables: Array of 11 observable values
    """
    import subprocess
    import tempfile
    
    # Path to Julia script
    julia_script = Path(__file__).parent / "j1j2_dmrg.jl"
    
    if not julia_script.exists():
        raise FileNotFoundError(f"Julia script not found: {julia_script}")
    
    # Create temp file for output
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_output = tmp.name
    
    try:
        # Run Julia DMRG
        cmd = [
            "julia",
            "--project=@.",
            str(julia_script),
            str(L),
            f"{j2_j1:.6f}",
            str(bond_dim),
            tmp_output
        ]
        
        logging.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per point
        )
        
        if result.returncode != 0:
            logging.error(f"Julia DMRG stdout:\n{result.stdout}")
            logging.error(f"Julia DMRG stderr:\n{result.stderr}")
            raise RuntimeError(f"Julia DMRG failed with code {result.returncode}")
        
        # Log Julia output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logging.info(f"[Julia] {line}")
        
        # Read results from temp HDF5
        with h5py.File(tmp_output, 'r') as f:
            energy = float(f['energy'][()])
            observables = f['observables'][:]
            psi = f['psi'][:]  # Dummy array for L>=6
        
        return psi, energy, observables
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_output):
            os.remove(tmp_output)


def run_quspin_fallback(
    L: int,
    j2_j1: float
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fallback to QuSpin for small lattices (L=4).
    
    Args:
        L: Lattice size (must be 4)
        j2_j1: Frustration ratio
        
    Returns:
        Tuple of (psi_array, energy, observables)
    """
    from src.ed_module import J1J2Hamiltonian
    from src.observable_module import ObservableModule
    from src.config import Config
    
    # Build Hamiltonian and compute ground state
    ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
    ham.build_hamiltonian()
    state = ham.compute_ground_state()
    
    # Compute observables
    # Load a minimal config for observable computation
    config = Config.from_yaml(project_root / "configs" / "default_config.yaml")
    obs_module = ObservableModule(config)
    obs_dict = obs_module.compute_all(state)
    
    # Convert to array in fixed order
    obs_names = [
        'energy', 'energy_density', 'staggered_magnetization', 
        'stripe_order', 'plaquette_order', 'S_pi_pi', 'S_pi_0',
        'entanglement_entropy', 'nematic_order', 'dimer_order_x', 'dimer_order_y'
    ]
    observables = np.array([obs_dict.get(name, 0.0) for name in obs_names])
    
    return state.coefficients, state.energy, observables


def save_groundstate_to_hdf5(
    filepath: Path,
    j2_val: float,
    L: int,
    psi: np.ndarray,
    energy: float,
    observables: np.ndarray,
    bond_dim: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save a single ground state to HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        j2_val: J2/J1 ratio
        L: Lattice size
        psi: Wavefunction coefficients
        energy: Ground state energy
        observables: Array of observable values
        bond_dim: Bond dimension used (for DMRG)
        metadata: Optional additional metadata
    """
    key = f"J2_{j2_val:.3f}"
    
    with h5py.File(filepath, 'a') as f:
        # Remove existing group if present
        if key in f:
            del f[key]
        
        grp = f.create_group(key)
        
        # Store wavefunction
        grp.create_dataset("psi", data=psi, compression='gzip', compression_opts=9)
        
        # Store energy as scalar dataset
        grp.create_dataset("energy", data=energy)
        
        # Store all 11 observables
        grp.create_dataset("observables", data=observables)
        
        # Store metadata as attributes
        grp.attrs["j2_j1"] = j2_val
        grp.attrs["L"] = L
        grp.attrs["chi"] = bond_dim
        grp.attrs["timestamp"] = datetime.now().isoformat()
        grp.attrs["hilbert_dim"] = len(psi)
        
        # Observable names for reference
        obs_names = [
            'energy', 'energy_density', 'staggered_magnetization',
            'stripe_order', 'plaquette_order', 'S_pi_pi', 'S_pi_0',
            'entanglement_entropy', 'nematic_order', 'dimer_order_x', 'dimer_order_y'
        ]
        grp.attrs["observable_names"] = obs_names
        
        if metadata:
            for k, v in metadata.items():
                try:
                    grp.attrs[k] = v
                except TypeError:
                    grp.attrs[k] = str(v)


def generate_groundstates(
    L: int,
    j2_range: np.ndarray,
    output_file: Path,
    bond_dim: int = 200,
    use_quspin: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    """Generate ground states for all J2 values with checkpointing.
    
    Args:
        L: Lattice size
        j2_range: Array of J2/J1 values to compute
        output_file: Path to output HDF5 file
        bond_dim: Maximum bond dimension for DMRG
        use_quspin: If True, use QuSpin instead of ITensor (L=4 only)
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check for already-completed points (checkpointing)
    completed = get_completed_points(output_file)
    logger.info(f"Found {len(completed)} already-completed points")
    
    n_total = len(j2_range)
    n_completed = 0
    
    for i, j2_val in enumerate(j2_range):
        key = f"J2_{j2_val:.3f}"
        
        # Skip if already done
        if key in completed:
            logger.info(f"Skipping J2={j2_val:.3f}, already done")
            n_completed += 1
            continue
        
        logger.info(f"Computing J2={j2_val:.3f} ({i+1}/{n_total})...")
        
        try:
            if use_quspin and L == 4:
                psi, energy, observables = run_quspin_fallback(L, j2_val)
                chi = 0  # Not applicable for exact diag
            else:
                psi, energy, observables = run_dmrg_itensor(
                    L=L,
                    j2_j1=j2_val,
                    bond_dim=bond_dim
                )
                chi = bond_dim
            
            # Save immediately (checkpoint)
            save_groundstate_to_hdf5(
                filepath=output_file,
                j2_val=j2_val,
                L=L,
                psi=psi,
                energy=energy,
                observables=observables,
                bond_dim=chi
            )
            
            n_completed += 1
            logger.info(
                f"  E={energy:.8f}, saved to {output_file.name} "
                f"({n_completed}/{n_total} complete)"
            )
            
        except Exception as e:
            logger.error(f"Failed for J2={j2_val:.3f}: {e}")
            raise
    
    logger.info(f"Generation complete: {n_completed}/{n_total} points saved to {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ground states for J1-J2 model using DMRG/ITensor"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--L",
        type=int,
        default=6,
        help="Lattice size (default: 6)"
    )
    parser.add_argument(
        "--j2_min",
        type=float,
        default=0.0,
        help="Minimum J2/J1 ratio (default: 0.0)"
    )
    parser.add_argument(
        "--j2_max",
        type=float,
        default=1.0,
        help="Maximum J2/J1 ratio (default: 1.0)"
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=41,
        help="Number of J2 points (default: 41)"
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=200,
        help="Maximum bond dimension for DMRG (default: 200)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 file path (default: groundstates_L{L}.h5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/groundstates",
        help="Output directory (default: results/groundstates)"
    )
    parser.add_argument(
        "--use_quspin",
        action="store_true",
        help="Use QuSpin instead of ITensor (L=4 only)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(output_dir)
    logger.info("=" * 60)
    logger.info("Ground State Generation for J1-J2 Heisenberg Model")
    logger.info("=" * 60)
    
    # Determine parameters
    L = args.L
    
    if args.config:
        config = Config.from_yaml(args.config)
        j2_range = config.get_j2_j1_values()
        logger.info(f"Loaded config from {args.config}")
    else:
        j2_range = np.linspace(args.j2_min, args.j2_max, args.n_points)
    
    # Output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = output_dir / f"groundstates_L{L}.h5"
    
    logger.info(f"Lattice size: L={L} ({L*L} spins)")
    logger.info(f"J2/J1 range: [{j2_range[0]:.3f}, {j2_range[-1]:.3f}] ({len(j2_range)} points)")
    logger.info(f"Bond dimension: {args.bond_dim}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Backend: {'QuSpin' if args.use_quspin else 'ITensor/DMRG'}")
    
    # Validate
    if args.use_quspin and L > 4:
        logger.error(f"QuSpin cannot handle L={L}. Use ITensor/DMRG instead.")
        sys.exit(1)
    
    # Generate ground states
    try:
        generate_groundstates(
            L=L,
            j2_range=j2_range,
            output_file=output_file,
            bond_dim=args.bond_dim,
            use_quspin=args.use_quspin,
            logger=logger
        )
    except NotImplementedError as e:
        logger.error(str(e))
        logger.info("For L=4, you can use --use_quspin flag to use the QuSpin backend.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
