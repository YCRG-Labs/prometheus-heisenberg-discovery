"""Exact Diagonalization Module for J1-J2 Heisenberg Model

This module implements exact diagonalization for computing ground states
of the frustrated J1-J2 Heisenberg model on square lattices using QuSpin.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
import time
import logging
from pathlib import Path
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import psutil

from src.exceptions import (
    ValidationError,
    NormalizationError,
    HermitianError,
    ConvergenceError,
    ComputationError
)


class J1J2Hamiltonian:
    """J1-J2 Heisenberg Hamiltonian on square lattice
    
    Constructs the Hamiltonian H = J1 Σ⟨i,j⟩ S⃗i·S⃗j + J2 Σ⟨⟨i,k⟩⟩ S⃗i·S⃗k
    where ⟨i,j⟩ denotes nearest neighbors and ⟨⟨i,k⟩⟩ denotes next-nearest neighbors.
    
    The Hamiltonian is constructed in the Sz=0 sector with translation symmetries
    for computational efficiency.
    
    Attributes:
        L: Linear lattice size (system has N = L×L spins)
        N: Total number of spins
        j2_j1: Frustration ratio J2/J1
        basis: QuSpin basis object with symmetries
        hamiltonian: Sparse Hamiltonian matrix in CSR format
    """
    
    def __init__(self, L: int, j2_j1: float):
        """Initialize J1-J2 Hamiltonian
        
        Args:
            L: Linear lattice size (must be in {4, 5, 6})
            j2_j1: Frustration ratio J2/J1 (typically in [0, 1])
            
        Raises:
            ValidationError: If L is not in valid range or j2_j1 is negative
        """
        if L not in {4, 5, 6}:
            raise ValidationError(
                f"Lattice size L={L} not supported",
                expected="{4, 5, 6}",
                actual=L,
                context={'parameter': 'L'}
            )
        if j2_j1 < 0:
            raise ValidationError(
                f"Frustration ratio j2_j1={j2_j1} must be non-negative",
                expected="j2_j1 >= 0",
                actual=j2_j1,
                context={'parameter': 'j2_j1'}
            )
        if not np.isfinite(j2_j1):
            raise ValidationError(
                f"Frustration ratio j2_j1={j2_j1} must be finite",
                expected="finite value",
                actual=j2_j1,
                context={'parameter': 'j2_j1'}
            )
            
        self.L = L
        self.N = L * L
        self.j2_j1 = j2_j1
        self.basis: Optional[Any] = None
        self.hamiltonian: Optional[csr_matrix] = None
        
    def _site_to_index(self, ix: int, iy: int) -> int:
        """Convert 2D lattice coordinates to 1D site index
        
        Args:
            ix: x-coordinate (0 to L-1)
            iy: y-coordinate (0 to L-1)
            
        Returns:
            Site index (0 to N-1)
        """
        return iy * self.L + ix
    
    def _get_nearest_neighbor_bonds(self) -> list:
        """Generate nearest-neighbor bond list
        
        Returns:
            List of [J1, i, j] bonds for all nearest-neighbor pairs
        """
        bonds = []
        J1 = 1.0  # Set J1 = 1 as energy scale
        
        for iy in range(self.L):
            for ix in range(self.L):
                i = self._site_to_index(ix, iy)
                
                # Right neighbor (periodic boundary conditions)
                j_right = self._site_to_index((ix + 1) % self.L, iy)
                bonds.append([J1, i, j_right])
                
                # Up neighbor (periodic boundary conditions)
                j_up = self._site_to_index(ix, (iy + 1) % self.L)
                bonds.append([J1, i, j_up])
                
        return bonds
    
    def _get_next_nearest_neighbor_bonds(self) -> list:
        """Generate next-nearest-neighbor bond list
        
        Returns:
            List of [J2, i, k] bonds for all next-nearest-neighbor pairs
        """
        bonds = []
        J2 = self.j2_j1  # J2 in units of J1
        
        for iy in range(self.L):
            for ix in range(self.L):
                i = self._site_to_index(ix, iy)
                
                # Diagonal neighbors (periodic boundary conditions)
                # Up-right diagonal
                k_ur = self._site_to_index((ix + 1) % self.L, (iy + 1) % self.L)
                bonds.append([J2, i, k_ur])
                
                # Up-left diagonal
                k_ul = self._site_to_index((ix - 1) % self.L, (iy + 1) % self.L)
                bonds.append([J2, i, k_ul])
                
        return bonds
    
    def build_hamiltonian(self) -> csr_matrix:
        """Construct Hamiltonian in Sz=0 sector with translation symmetries
        
        Builds the sparse Hamiltonian matrix using QuSpin operators.
        The spin interaction S⃗i·S⃗j is decomposed as:
        S⃗i·S⃗j = Sᶻᵢ Sᶻⱼ + ½(S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ)
        
        Returns:
            Sparse Hamiltonian matrix in CSR format
            
        Raises:
            ComputationError: If Hamiltonian construction fails
            HermitianError: If constructed Hamiltonian is not Hermitian
        """
        try:
            # Create basis in Sz=0 sector with translation symmetries
            # For spin-1/2, Sz=0 means N_up = N_down = N/2
            if self.N % 2 != 0:
                raise ValidationError(
                    f"Cannot construct Sz=0 sector for odd N={self.N}",
                    expected="even N",
                    actual=self.N,
                    context={'L': self.L, 'N': self.N}
                )
            
            # Create basis in Sz=0 sector without translation symmetries for simplicity
            # Translation symmetries can be added later for optimization
            self.basis = spin_basis_general(
                N=self.N,
                Nup=self.N // 2  # Sz=0 sector
            )
            
            # Get bond lists
            nn_bonds = self._get_nearest_neighbor_bonds()
            nnn_bonds = self._get_next_nearest_neighbor_bonds()
            
            # Define operator strings for S⃗i·S⃗j = Sᶻᵢ Sᶻⱼ + ½(S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ)
            # QuSpin notation: "zz" for Sz_i Sz_j, "+-" for S+_i S-_j, "-+" for S-_i S+_j
            static_list = [
                ["zz", nn_bonds],      # J1 * Sz_i Sz_j for NN
                ["+-", [[0.5 * J, i, j] for J, i, j in nn_bonds]],  # 0.5 * J1 * S+_i S-_j
                ["-+", [[0.5 * J, i, j] for J, i, j in nn_bonds]],  # 0.5 * J1 * S-_i S+_j
            ]
            
            # Add next-nearest-neighbor terms if J2 != 0
            if abs(self.j2_j1) > 1e-12:
                static_list.extend([
                    ["zz", nnn_bonds],     # J2 * Sz_i Sz_k for NNN
                    ["+-", [[0.5 * J, i, k] for J, i, k in nnn_bonds]],  # 0.5 * J2 * S+_i S-_k
                    ["-+", [[0.5 * J, i, k] for J, i, k in nnn_bonds]],  # 0.5 * J2 * S-_i S+_k
                ])
            
            # Build Hamiltonian (no time-dependent terms)
            H = hamiltonian(
                static_list,
                [],  # No dynamic terms
                basis=self.basis,
                dtype=np.float64,
                check_herm=True,  # Verify Hermiticity
                check_symm=False  # Don't check symmetries (we trust our construction)
            )
            
            # Store as CSR sparse matrix
            self.hamiltonian = H.tocsr()
            
            # Verify Hermiticity explicitly
            if not self.verify_hermiticity(tol=1e-8):
                H_dag = self.hamiltonian.conj().transpose()
                diff = self.hamiltonian - H_dag
                max_deviation = float(np.max(np.abs(diff.data)))
                raise HermitianError(
                    "Constructed Hamiltonian is not Hermitian",
                    max_deviation=max_deviation,
                    tolerance=1e-8,
                    context={'L': self.L, 'j2_j1': self.j2_j1}
                )
            
            return self.hamiltonian
            
        except (ValidationError, HermitianError):
            raise
        except Exception as e:
            raise ComputationError(
                f"Failed to construct Hamiltonian",
                context={'L': self.L, 'j2_j1': self.j2_j1, 'error': str(e)}
            ) from e
    
    def get_hilbert_dim(self) -> int:
        """Get dimension of Hilbert space in Sz=0 sector with symmetries
        
        Returns:
            Dimension of the symmetry-reduced Hilbert space
            
        Raises:
            RuntimeError: If basis has not been constructed
        """
        if self.basis is None:
            raise RuntimeError("Basis not constructed. Call build_hamiltonian() first.")
        return self.basis.Ns
    
    def verify_hermiticity(self, tol: float = 1e-8) -> bool:
        """Verify that the Hamiltonian is Hermitian
        
        Args:
            tol: Tolerance for Hermiticity check
            
        Returns:
            True if ||H - H†|| < tol, False otherwise
            
        Raises:
            ComputationError: If Hamiltonian has not been constructed
        """
        if self.hamiltonian is None:
            raise ComputationError(
                "Hamiltonian not constructed",
                context={'operation': 'verify_hermiticity', 'L': self.L, 'j2_j1': self.j2_j1}
            )
        
        # Compute ||H - H†||_F (Frobenius norm)
        H_dag = self.hamiltonian.conj().transpose()
        diff = self.hamiltonian - H_dag
        norm_diff = np.sqrt(diff.multiply(diff.conj()).sum())
        
        return norm_diff < tol
    
    def compute_ground_state(self, tol: float = 1e-10, maxiter: int = 1000) -> 'GroundState':
        """Compute ground state using Lanczos algorithm
        
        Uses scipy's eigsh (Lanczos algorithm) to find the lowest eigenvalue
        and eigenvector of the Hamiltonian. Monitors convergence and validates
        the result.
        
        Args:
            tol: Convergence tolerance for energy (default: 1e-10)
            maxiter: Maximum number of Lanczos iterations (default: 1000)
            
        Returns:
            GroundState object containing wavefunction and energy
            
        Raises:
            ComputationError: If Hamiltonian not constructed
            ConvergenceError: If Lanczos algorithm fails to converge
            NormalizationError: If ground state is not properly normalized
        """
        if self.hamiltonian is None:
            raise ComputationError(
                "Hamiltonian not constructed",
                context={'operation': 'compute_ground_state', 'L': self.L, 'j2_j1': self.j2_j1}
            )
        
        start_time = time.time()
        
        try:
            # Use Lanczos algorithm to find lowest eigenvalue
            # which='SA' means smallest algebraic eigenvalue
            # k=1 means we want only the ground state
            eigenvalues, eigenvectors = eigsh(
                self.hamiltonian,
                k=1,
                which='SA',
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=True
            )
            
            ground_energy = float(eigenvalues[0])
            ground_state_coeffs = eigenvectors[:, 0]
            
            # Check for NaN or Inf in results
            if not np.all(np.isfinite(ground_state_coeffs)):
                raise ComputationError(
                    "Ground state contains NaN or Inf values",
                    context={'L': self.L, 'j2_j1': self.j2_j1, 'energy': ground_energy}
                )
            
            if not np.isfinite(ground_energy):
                raise ComputationError(
                    "Ground state energy is NaN or Inf",
                    context={'L': self.L, 'j2_j1': self.j2_j1, 'energy': ground_energy}
                )
            
            # Ensure proper normalization
            norm = np.sqrt(np.real(np.vdot(ground_state_coeffs, ground_state_coeffs)))
            if abs(norm - 1.0) > 1e-8:
                # Renormalize if needed
                ground_state_coeffs = ground_state_coeffs / norm
            
            # Verify normalization after correction
            final_norm = np.sqrt(np.real(np.vdot(ground_state_coeffs, ground_state_coeffs)))
            if abs(final_norm - 1.0) > 1e-8:
                raise NormalizationError(
                    "Ground state normalization failed",
                    norm=final_norm,
                    tolerance=1e-8,
                    context={'L': self.L, 'j2_j1': self.j2_j1}
                )
            
            computation_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                'convergence_tol': tol,
                'maxiter': maxiter,
                'computation_time': computation_time,
                'hilbert_dim': self.basis.Ns,
                'converged': True
            }
            
            # Create GroundState object
            ground_state = GroundState(
                coefficients=ground_state_coeffs,
                energy=ground_energy,
                basis=self.basis,
                j2_j1=self.j2_j1,
                L=self.L,
                metadata=metadata
            )
            
            # Validate the ground state
            if not ground_state.validate(tol=1e-8):
                raise NormalizationError(
                    "Ground state validation failed",
                    norm=ground_state.norm(),
                    tolerance=1e-8,
                    context={'L': self.L, 'j2_j1': self.j2_j1}
                )
            
            return ground_state
            
        except (ComputationError, NormalizationError):
            raise
        except np.linalg.LinAlgError as e:
            raise ConvergenceError(
                f"Lanczos algorithm failed to converge",
                iterations=maxiter,
                context={'L': self.L, 'j2_j1': self.j2_j1, 'error': str(e)}
            ) from e
        except Exception as e:
            raise ComputationError(
                f"Ground state computation failed",
                context={'L': self.L, 'j2_j1': self.j2_j1, 'error': str(e)}
            ) from e



class GroundState:
    """Represents a quantum ground state wavefunction
    
    Stores the ground state wavefunction coefficients along with metadata
    about the system parameters and energy. Provides methods for validation
    and conversion to formats suitable for machine learning.
    
    Attributes:
        coefficients: Complex array of wavefunction coefficients in the basis
        energy: Ground state energy
        basis: QuSpin basis object
        j2_j1: Frustration ratio parameter
        L: Lattice size
        metadata: Additional information (convergence, timing, etc.)
    """
    
    def __init__(
        self,
        coefficients: np.ndarray,
        energy: float,
        basis: Any,
        j2_j1: float,
        L: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize GroundState
        
        Args:
            coefficients: Complex wavefunction coefficients
            energy: Ground state energy
            basis: QuSpin basis object
            j2_j1: Frustration ratio
            L: Lattice size
            metadata: Optional metadata dictionary
            
        Raises:
            ValidationError: If coefficients are not 1D array or contain invalid values
        """
        if coefficients.ndim != 1:
            raise ValidationError(
                f"Coefficients must be 1D array",
                expected="1D array",
                actual=f"{coefficients.ndim}D array",
                context={'shape': coefficients.shape}
            )
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(coefficients)):
            raise ValidationError(
                "Coefficients contain NaN or Inf values",
                expected="finite values",
                actual="NaN or Inf",
                context={'L': L, 'j2_j1': j2_j1}
            )
        
        # Check energy is finite
        if not np.isfinite(energy):
            raise ValidationError(
                "Energy is NaN or Inf",
                expected="finite value",
                actual=energy,
                context={'L': L, 'j2_j1': j2_j1}
            )
        
        # Ensure coefficients are complex
        if not np.iscomplexobj(coefficients):
            coefficients = coefficients.astype(np.complex128)
        
        self.coefficients = coefficients
        self.energy = energy
        self.basis = basis
        self.j2_j1 = j2_j1
        self.L = L
        self.metadata = metadata or {}
        
    def to_real_vector(self) -> np.ndarray:
        """Convert complex wavefunction to real vector for Q-VAE input
        
        Concatenates real and imaginary parts: [Re(c), Im(c)]
        This preserves all quantum information in a real-valued format
        suitable for neural network processing.
        
        Returns:
            Real vector of length 2 * dim(H) containing [Re(c), Im(c)]
        """
        real_part = np.real(self.coefficients)
        imag_part = np.imag(self.coefficients)
        return np.concatenate([real_part, imag_part])
    
    def norm(self) -> float:
        """Compute wavefunction norm ⟨ψ|ψ⟩
        
        Returns:
            Norm of the wavefunction (should be 1.0 for normalized states)
        """
        return float(np.real(np.vdot(self.coefficients, self.coefficients)))
    
    def validate(self, tol: float = 1e-8) -> bool:
        """Validate wavefunction normalization and consistency
        
        Checks that:
        1. Wavefunction is normalized: |⟨ψ|ψ⟩ - 1| < tol
        2. Coefficients are finite (no NaN or Inf)
        3. Dimension matches basis dimension
        
        Args:
            tol: Tolerance for normalization check
            
        Returns:
            True if all checks pass, False otherwise
        """
        # Check normalization
        norm_val = self.norm()
        if abs(norm_val - 1.0) >= tol:
            return False
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(self.coefficients)):
            return False
        
        # Check dimension consistency
        if len(self.coefficients) != self.basis.Ns:
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of GroundState"""
        return (
            f"GroundState(L={self.L}, j2_j1={self.j2_j1:.4f}, "
            f"E={self.energy:.6f}, dim={len(self.coefficients)})"
        )



class EDModule:
    """Exact Diagonalization Module for parameter sweeps
    
    Manages computation of ground states across multiple parameter points
    (j2_j1, L) with support for parallel execution and checkpointing.
    
    Attributes:
        config: Configuration object with ED parameters
        logger: Logger for progress tracking
        checkpoint_dir: Directory for saving checkpoints
    """
    
    def __init__(self, config: Any, checkpoint_dir: Optional[Path] = None):
        """Initialize EDModule
        
        Args:
            config: Configuration object with ed_parameters attribute
            checkpoint_dir: Optional directory for checkpoints (default: config.paths.checkpoint_dir)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if checkpoint_dir is None:
            self.checkpoint_dir = Path(config.paths.checkpoint_dir) / "ed_checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory monitoring
        self.monitor_memory = getattr(config.ed_parameters, 'monitor_memory', True)
        self.clear_cache = getattr(config.ed_parameters, 'clear_cache_after_computation', True)
        self.process = psutil.Process() if self.monitor_memory else None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB
        
        Returns:
            Memory usage in GB
        """
        if not self.monitor_memory or self.process is None:
            return 0.0
        
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 ** 3)  # Convert bytes to GB
        except Exception:
            return 0.0
    
    def _log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage
        
        Args:
            context: Context string for logging
        """
        if not self.monitor_memory:
            return
        
        mem_usage = self._get_memory_usage()
        if context:
            self.logger.debug(f"Memory usage ({context}): {mem_usage:.2f} GB")
        else:
            self.logger.debug(f"Memory usage: {mem_usage:.2f} GB")
    
    def _clear_memory_cache(self) -> None:
        """Clear memory cache and run garbage collection"""
        if not self.clear_cache:
            return
        
        # Run garbage collection
        gc.collect()
        
        self.logger.debug("Memory cache cleared")
        
    def _compute_single_point(
        self,
        j2_j1: float,
        L: int,
        tol: float = 1e-10
    ) -> Tuple[Tuple[float, int], GroundState]:
        """Compute ground state for a single parameter point
        
        Args:
            j2_j1: Frustration ratio
            L: Lattice size
            tol: Convergence tolerance
            
        Returns:
            Tuple of ((j2_j1, L), GroundState)
        """
        try:
            # Log memory before computation
            self._log_memory_usage(f"before L={L}, j2_j1={j2_j1:.4f}")
            
            # Create Hamiltonian
            ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
            ham.build_hamiltonian()
            
            # Log memory after Hamiltonian construction
            self._log_memory_usage(f"after Hamiltonian L={L}, j2_j1={j2_j1:.4f}")
            
            # Compute ground state
            ground_state = ham.compute_ground_state(tol=tol)
            
            # Log memory after ground state computation
            self._log_memory_usage(f"after ground state L={L}, j2_j1={j2_j1:.4f}")
            
            self.logger.info(
                f"Computed ground state: L={L}, j2_j1={j2_j1:.4f}, "
                f"E={ground_state.energy:.6f}, dim={len(ground_state.coefficients)}"
            )
            
            # Clear Hamiltonian from memory
            del ham
            self._clear_memory_cache()
            
            return ((j2_j1, L), ground_state)
            
        except Exception as e:
            self.logger.error(
                f"Failed to compute ground state for L={L}, j2_j1={j2_j1:.4f}: {e}"
            )
            raise
    
    def _get_checkpoint_path(self, L: int) -> Path:
        """Get checkpoint file path for a given lattice size
        
        Args:
            L: Lattice size
            
        Returns:
            Path to checkpoint file
        """
        return self.checkpoint_dir / f"ed_checkpoint_L{L}.pkl"
    
    def _save_checkpoint(
        self,
        results: Dict[Tuple[float, int], GroundState],
        L: int
    ) -> None:
        """Save checkpoint for a lattice size
        
        Args:
            results: Dictionary of computed ground states
            L: Lattice size
        """
        checkpoint_path = self._get_checkpoint_path(L)
        
        # Filter results for this lattice size
        L_results = {k: v for k, v in results.items() if k[1] == L}
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(L_results, f)
        
        self.logger.info(f"Saved checkpoint for L={L} with {len(L_results)} states")
    
    def _load_checkpoint(self, L: int) -> Dict[Tuple[float, int], GroundState]:
        """Load checkpoint for a lattice size
        
        Args:
            L: Lattice size
            
        Returns:
            Dictionary of ground states from checkpoint, or empty dict if no checkpoint
        """
        checkpoint_path = self._get_checkpoint_path(L)
        
        if not checkpoint_path.exists():
            return {}
        
        try:
            with open(checkpoint_path, 'rb') as f:
                results = pickle.load(f)
            self.logger.info(f"Loaded checkpoint for L={L} with {len(results)} states")
            return results
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint for L={L}: {e}")
            return {}
    
    def run_parameter_sweep(
        self,
        parallel: Optional[bool] = None,
        n_processes: Optional[int] = None,
        resume: bool = True
    ) -> Dict[Tuple[float, int], GroundState]:
        """Compute ground states for all (j2_j1, L) parameter points
        
        Performs a complete parameter sweep over all j2_j1 values and lattice
        sizes specified in the configuration. Supports parallel execution and
        checkpointing for resumption.
        
        Args:
            parallel: Whether to use parallel execution (default: from config)
            n_processes: Number of processes for parallel execution (default: from config or cpu_count())
            resume: Whether to resume from checkpoints (default: True)
            
        Returns:
            Dictionary mapping (j2_j1, L) -> GroundState for all parameter points
        """
        # Use config values if not specified
        if parallel is None:
            parallel = getattr(self.config.ed_parameters, 'parallel', True)
        if n_processes is None:
            n_processes = getattr(self.config.ed_parameters, 'n_processes', None)
        
        # Get parameter points from config
        j2_j1_values = self.config.get_j2_j1_values()
        lattice_sizes = self.config.ed_parameters.lattice_sizes
        tol = self.config.ed_parameters.lanczos_tol
        
        # Initialize results dictionary
        results: Dict[Tuple[float, int], GroundState] = {}
        
        # Load checkpoints if resuming
        if resume:
            for L in lattice_sizes:
                checkpoint_results = self._load_checkpoint(L)
                results.update(checkpoint_results)
        
        # Determine which points need to be computed
        all_points = [
            (j2_j1, L)
            for j2_j1 in j2_j1_values
            for L in lattice_sizes
        ]
        
        points_to_compute = [
            point for point in all_points
            if point not in results
        ]
        
        if not points_to_compute:
            self.logger.info("All parameter points already computed")
            return results
        
        self.logger.info(
            f"Computing {len(points_to_compute)} parameter points "
            f"({len(results)} already computed)"
        )
        
        # Compute ground states
        if parallel and len(points_to_compute) > 1:
            # Parallel execution
            if n_processes is None:
                n_processes = min(cpu_count(), len(points_to_compute))
            
            self.logger.info(f"Using {n_processes} parallel processes")
            
            # Create partial function with fixed tolerance
            compute_func = partial(
                self._compute_single_point_wrapper,
                tol=tol
            )
            
            n_total = len(points_to_compute)
            completed = 0
            with Pool(processes=n_processes) as pool:
                for key, state in pool.imap_unordered(compute_func, points_to_compute, chunksize=1):
                    results[key] = state
                    completed += 1
                    j2_j1, L = key
                    self.logger.info(
                        f"ED progress: {completed}/{n_total} — L={L}, j2_j1={j2_j1:.4f}, E={state.energy:.6f}"
                    )
                    # Save checkpoint when all points for this L are done
                    L_points = [p for p in all_points if p[1] == L]
                    L_computed = [p for p in L_points if p in results]
                    if len(L_computed) == len(L_points):
                        self._save_checkpoint(results, L)
                        self.logger.info(f"Checkpoint saved for L={L} ({len(L_computed)} states)")
        else:
            # Sequential execution
            for j2_j1, L in points_to_compute:
                key, state = self._compute_single_point(j2_j1, L, tol)
                results[key] = state
                
                # Save checkpoint after each lattice size is complete
                L_points = [p for p in all_points if p[1] == L]
                L_computed = [p for p in L_points if p in results]
                
                if len(L_computed) == len(L_points):
                    self._save_checkpoint(results, L)
        
        self.logger.info(f"Parameter sweep complete: {len(results)} states computed")
        
        return results
    
    @staticmethod
    def _compute_single_point_wrapper(
        point: Tuple[float, int],
        tol: float
    ) -> Tuple[Tuple[float, int], GroundState]:
        """Wrapper for _compute_single_point for use with multiprocessing
        
        Args:
            point: (j2_j1, L) tuple
            tol: Convergence tolerance
            
        Returns:
            Tuple of ((j2_j1, L), GroundState)
        """
        j2_j1, L = point
        
        # Create a temporary EDModule instance for this process
        # (can't pickle logger, so create minimal version)
        ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
        ham.build_hamiltonian()
        ground_state = ham.compute_ground_state(tol=tol)
        
        return ((j2_j1, L), ground_state)
