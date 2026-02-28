"""Data Storage Module for J1-J2 Heisenberg Prometheus Framework

This module implements persistent storage of all computational results with metadata
using HDF5 for efficient storage of large arrays and JSON for metadata.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import h5py
import json
import torch
import logging
from datetime import datetime
import sys


class DataStorage:
    """Persistent storage manager for computational results
    
    Manages storage of ground states, observables, latent representations,
    and Q-VAE models using HDF5 format with comprehensive metadata tracking.
    
    Attributes:
        data_dir: Directory for data storage
        hdf5_file: Path to main HDF5 file
        logger: Logger for tracking operations
    """
    
    def __init__(self, config: Any):
        """Initialize DataStorage
        
        Args:
            config: Configuration object with paths attribute
        """
        self.data_dir = Path(config.paths.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.hdf5_file = self.data_dir / "j1j2_data.h5"
        self.logger = logging.getLogger(__name__)
        
        # Initialize HDF5 file structure if it doesn't exist
        self._initialize_hdf5_structure()
        
    def _initialize_hdf5_structure(self) -> None:
        """Initialize HDF5 file with group structure"""
        with h5py.File(self.hdf5_file, 'a') as f:
            # Create main groups if they don't exist
            if 'ground_states' not in f:
                f.create_group('ground_states')
            if 'observables' not in f:
                f.create_group('observables')
            if 'latent_representations' not in f:
                f.create_group('latent_representations')
            if 'metadata' not in f:
                metadata_group = f.create_group('metadata')
                # Store software versions (convert to strings explicitly)
                metadata_group.attrs['numpy_version'] = str(np.__version__)
                metadata_group.attrs['pandas_version'] = str(pd.__version__)
                metadata_group.attrs['h5py_version'] = str(h5py.__version__)
                metadata_group.attrs['torch_version'] = str(torch.__version__)
                metadata_group.attrs['python_version'] = str(sys.version)
                metadata_group.attrs['creation_time'] = str(datetime.now().isoformat())
    
    def _get_ground_state_key(self, j2_j1: float, L: int) -> str:
        """Generate HDF5 key for ground state
        
        Args:
            j2_j1: Frustration ratio
            L: Lattice size
            
        Returns:
            String key for HDF5 dataset
        """
        # Format: "L{L}_j2j1_{j2_j1:.6f}"
        return f"L{L}_j2j1_{j2_j1:.6f}"
    
    def save_ground_state(
        self,
        state: Any,  # GroundState object
        j2_j1: float,
        L: int
    ) -> None:
        """Save ground state wavefunction and metadata to HDF5
        
        Stores the complex wavefunction coefficients, energy, and all metadata
        associated with the ground state computation.
        
        Args:
            state: GroundState object to save
            j2_j1: Frustration ratio
            L: Lattice size
            
        Raises:
            ValueError: If state is not properly normalized
            IOError: If HDF5 write fails
        """
        # Validate normalization (skip basis check if basis is None)
        norm_val = state.norm()
        if abs(norm_val - 1.0) >= 1e-8:
            raise ValueError(
                f"Cannot save unnormalized ground state for L={L}, j2_j1={j2_j1}: norm={norm_val}"
            )
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(state.coefficients)):
            raise ValueError(
                f"Cannot save ground state with NaN/Inf for L={L}, j2_j1={j2_j1}"
            )
        
        key = self._get_ground_state_key(j2_j1, L)
        
        try:
            with h5py.File(self.hdf5_file, 'a') as f:
                gs_group = f['ground_states']
                
                # Remove existing dataset if present
                if key in gs_group:
                    del gs_group[key]
                
                # Create dataset for wavefunction coefficients
                # Store as complex128
                dataset = gs_group.create_dataset(
                    key,
                    data=state.coefficients,
                    dtype=np.complex128,
                    compression='gzip',
                    compression_opts=9
                )
                
                # Store metadata as attributes
                dataset.attrs['energy'] = state.energy
                dataset.attrs['j2_j1'] = j2_j1
                dataset.attrs['L'] = L
                dataset.attrs['N'] = L * L
                dataset.attrs['hilbert_dim'] = len(state.coefficients)
                dataset.attrs['norm'] = state.norm()
                dataset.attrs['save_time'] = datetime.now().isoformat()
                
                # Store computation metadata
                for key_meta, value in state.metadata.items():
                    # Convert to JSON-serializable types
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    dataset.attrs[f'meta_{key_meta}'] = value
            
            self.logger.info(
                f"Saved ground state: L={L}, j2_j1={j2_j1:.4f}, "
                f"dim={len(state.coefficients)}"
            )
            
        except Exception as e:
            raise IOError(
                f"Failed to save ground state for L={L}, j2_j1={j2_j1}: {e}"
            ) from e
    
    def load_ground_state(
        self,
        j2_j1: float,
        L: int
    ) -> Any:  # Returns GroundState object
        """Load ground state from HDF5 storage
        
        Retrieves the wavefunction coefficients and metadata for a specific
        parameter point.
        
        Args:
            j2_j1: Frustration ratio
            L: Lattice size
            
        Returns:
            GroundState object
            
        Raises:
            KeyError: If ground state not found in storage
            IOError: If HDF5 read fails
        """
        from src.ed_module import GroundState
        
        key = self._get_ground_state_key(j2_j1, L)
        
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                gs_group = f['ground_states']
                
                if key not in gs_group:
                    raise KeyError(
                        f"Ground state not found for L={L}, j2_j1={j2_j1}"
                    )
                
                dataset = gs_group[key]
                
                # Load wavefunction coefficients
                coefficients = dataset[:]
                
                # Load metadata
                energy = float(dataset.attrs['energy'])
                
                # Reconstruct metadata dictionary
                metadata = {}
                for attr_name in dataset.attrs:
                    if attr_name.startswith('meta_'):
                        key_name = attr_name[5:]  # Remove 'meta_' prefix
                        metadata[key_name] = dataset.attrs[attr_name]
                
                # Create GroundState object
                # Note: basis is not stored, set to None
                ground_state = GroundState(
                    coefficients=coefficients,
                    energy=energy,
                    basis=None,  # Basis not stored
                    j2_j1=j2_j1,
                    L=L,
                    metadata=metadata
                )
                
                self.logger.info(
                    f"Loaded ground state: L={L}, j2_j1={j2_j1:.4f}, "
                    f"dim={len(coefficients)}"
                )
                
                return ground_state
                
        except KeyError:
            raise
        except Exception as e:
            raise IOError(
                f"Failed to load ground state for L={L}, j2_j1={j2_j1}: {e}"
            ) from e
    
    def load_ground_states_for_lattice_size(self, L: int) -> Dict[Tuple[int, float], Any]:
        """Load all ground states for a specific lattice size
        
        Args:
            L: Lattice size
            
        Returns:
            Dictionary mapping (L, j2_j1) tuples to GroundState objects
            
        Raises:
            IOError: If HDF5 read fails
        """
        states = {}
        
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                gs_group = f['ground_states']
                
                # Find all keys for this lattice size
                prefix = f"L{L}_j2j1_"
                for key in gs_group.keys():
                    if key.startswith(prefix):
                        # Extract j2_j1 from key
                        j2_j1_str = key[len(prefix):]
                        j2_j1 = float(j2_j1_str)
                        
                        # Load this ground state
                        state = self.load_ground_state(j2_j1, L)
                        states[(L, j2_j1)] = state
            
            self.logger.info(f"Loaded {len(states)} ground states for L={L}")
            return states
            
        except Exception as e:
            raise IOError(
                f"Failed to load ground states for L={L}: {e}"
            ) from e
    
    def save_observables(self, observables: pd.DataFrame) -> None:
        """Save observable DataFrame to CSV (simpler than HDF5 for mixed types)
        
        Stores all computed observables in CSV format for easy access.
        
        Args:
            observables: DataFrame with columns including j2_j1, L, and observable values
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
            IOError: If CSV write fails
        """
        if observables.empty:
            raise ValueError("Cannot save empty observables DataFrame")
        
        # Validate required columns
        required_cols = ['j2_j1', 'L']
        missing_cols = [col for col in required_cols if col not in observables.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        try:
            # Save to CSV in output directory
            csv_file = self.output_dir / "observables.csv"
            observables.to_csv(csv_file, index=False)
            
            self.logger.info(
                f"Saved observables to CSV: {len(observables)} rows, "
                f"{len(observables.columns)} columns"
            )
            
        except Exception as e:
            raise IOError(f"Failed to save observables: {e}") from e
    
    def load_observables(self) -> pd.DataFrame:
        """Load observables DataFrame from CSV
        
        Returns:
            DataFrame with all computed observables
            
        Raises:
            FileNotFoundError: If observables CSV not found
            IOError: If CSV read fails
        """
        try:
            csv_file = self.output_dir / "observables.csv"
            
            if not csv_file.exists():
                raise FileNotFoundError(f"Observables CSV not found: {csv_file}")
            
            observables = pd.read_csv(csv_file)
            
            self.logger.info(
                f"Loaded observables from CSV: {len(observables)} rows, "
                f"{len(observables.columns)} columns"
            )
            
            return observables
                
        except FileNotFoundError:
            raise
        except Exception as e:
            raise IOError(f"Failed to load observables: {e}") from e
    
    def save_latent_representations(
        self,
        latent_reps: Dict[Tuple[float, int], np.ndarray]
    ) -> None:
        """Save latent representations to HDF5
        
        Stores the latent space encodings for all ground states, indexed by
        (j2_j1, L) parameter points.
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            
        Raises:
            ValueError: If latent_reps is empty
            IOError: If HDF5 write fails
        """
        if not latent_reps:
            raise ValueError("Cannot save empty latent representations")
        
        try:
            with h5py.File(self.hdf5_file, 'a') as f:
                latent_group = f['latent_representations']
                
                # Clear existing data
                for key in list(latent_group.keys()):
                    del latent_group[key]
                
                # Save each latent representation
                for (j2_j1, L), z in latent_reps.items():
                    key = self._get_ground_state_key(j2_j1, L)
                    
                    dataset = latent_group.create_dataset(
                        key,
                        data=z,
                        dtype=np.float64,
                        compression='gzip',
                        compression_opts=9
                    )
                    
                    # Store metadata
                    dataset.attrs['j2_j1'] = j2_j1
                    dataset.attrs['L'] = L
                    dataset.attrs['latent_dim'] = len(z)
                    dataset.attrs['save_time'] = datetime.now().isoformat()
                
                # Store summary metadata
                latent_group.attrs['n_representations'] = len(latent_reps)
                latent_group.attrs['save_time'] = datetime.now().isoformat()
            
            self.logger.info(
                f"Saved {len(latent_reps)} latent representations"
            )
            
        except Exception as e:
            raise IOError(f"Failed to save latent representations: {e}") from e
    
    def load_latent_representations(
        self
    ) -> Dict[Tuple[float, int], np.ndarray]:
        """Load latent representations from HDF5
        
        Returns:
            Dictionary mapping (j2_j1, L) -> latent vector
            
        Raises:
            KeyError: If latent representations not found
            IOError: If HDF5 read fails
        """
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                if 'latent_representations' not in f:
                    raise KeyError("Latent representations not found in storage")
                
                latent_group = f['latent_representations']
                
                if len(latent_group.keys()) == 0:
                    raise KeyError("No latent representations found in storage")
                
                latent_reps = {}
                
                for key in latent_group.keys():
                    dataset = latent_group[key]
                    
                    # Load latent vector
                    z = dataset[:]
                    
                    # Load metadata
                    j2_j1 = float(dataset.attrs['j2_j1'])
                    L = int(dataset.attrs['L'])
                    
                    latent_reps[(j2_j1, L)] = z
                
                self.logger.info(
                    f"Loaded {len(latent_reps)} latent representations"
                )
                
                return latent_reps
                
        except KeyError:
            raise
        except Exception as e:
            raise IOError(f"Failed to load latent representations: {e}") from e
    
    def save_qvae_model(
        self,
        model: Any,  # QVAE model
        L: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save Q-VAE model checkpoint
        
        Stores the model state dict and associated metadata for a specific
        lattice size.
        
        Args:
            model: QVAE model to save
            L: Lattice size this model was trained for
            metadata: Optional metadata (hyperparameters, training history, etc.)
            
        Raises:
            IOError: If checkpoint save fails
        """
        checkpoint_dir = self.data_dir / "qvae_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"qvae_L{L}.pt"
        
        try:
            # Prepare checkpoint dictionary
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'L': L,
                'save_time': datetime.now().isoformat(),
            }
            
            # Add metadata if provided
            if metadata is not None:
                checkpoint['metadata'] = metadata
            
            # Add software versions for reproducibility
            checkpoint['versions'] = {
                'torch': torch.__version__,
                'numpy': np.__version__,
                'python': sys.version,
            }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            self.logger.info(
                f"Saved Q-VAE model checkpoint for L={L} to {checkpoint_path}"
            )
            
        except Exception as e:
            raise IOError(
                f"Failed to save Q-VAE model for L={L}: {e}"
            ) from e
    
    def load_qvae_model(
        self,
        model: Any,  # QVAE model
        L: int
    ) -> Dict[str, Any]:
        """Load Q-VAE model checkpoint
        
        Loads the model state dict and metadata for a specific lattice size.
        
        Args:
            model: QVAE model to load state into
            L: Lattice size
            
        Returns:
            Dictionary containing metadata from checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint not found
            IOError: If checkpoint load fails
        """
        checkpoint_dir = self.data_dir / "qvae_checkpoints"
        checkpoint_path = checkpoint_dir / f"qvae_L{L}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Q-VAE checkpoint not found for L={L} at {checkpoint_path}"
            )
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Extract metadata
            metadata = checkpoint.get('metadata', {})
            metadata['L'] = checkpoint.get('L', L)
            metadata['save_time'] = checkpoint.get('save_time', 'unknown')
            metadata['versions'] = checkpoint.get('versions', {})
            
            self.logger.info(
                f"Loaded Q-VAE model checkpoint for L={L} from {checkpoint_path}"
            )
            
            return metadata
            
        except Exception as e:
            raise IOError(
                f"Failed to load Q-VAE model for L={L}: {e}"
            ) from e
    
    def save_analysis_results(
        self,
        data: Any,
        filename: str
    ) -> None:
        """Save analysis results (e.g. order params, critical points) to JSON in output_dir.

        Args:
            data: Serializable dict/list (tuples converted to lists for JSON).
            filename: Filename under output_dir (e.g. 'critical_points.json').
        """
        out = self.output_dir / filename
        try:
            def _to_serializable(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: _to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_serializable(x) for x in obj]
                if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                    return _to_serializable(obj.to_dict())
                if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if hasattr(obj, "__dict__"):
                    # Fallback for simple result/validation objects
                    return _to_serializable(vars(obj))
                return obj
            with open(out, 'w') as f:
                json.dump(_to_serializable(data), f, indent=2)
            self.logger.info(f"Saved analysis results: {out}")
        except Exception as e:
            raise IOError(f"Failed to save analysis results to {out}: {e}") from e

    def save_metadata(
        self,
        key: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Save arbitrary metadata to HDF5
        
        Stores metadata in JSON format under a specified key.
        
        Args:
            key: Metadata key/identifier
            metadata: Dictionary of metadata to save
            
        Raises:
            IOError: If metadata save fails
        """
        try:
            with h5py.File(self.hdf5_file, 'a') as f:
                metadata_group = f['metadata']
                
                # Convert metadata to JSON string
                metadata_json = json.dumps(metadata, indent=2)
                
                # Store as string dataset
                if key in metadata_group:
                    del metadata_group[key]
                
                dataset = metadata_group.create_dataset(
                    key,
                    data=metadata_json,
                    dtype=h5py.string_dtype()
                )
                
                dataset.attrs['save_time'] = datetime.now().isoformat()
            
            self.logger.info(f"Saved metadata: {key}")
            
        except Exception as e:
            raise IOError(f"Failed to save metadata '{key}': {e}") from e
    
    def load_metadata(self, key: str) -> Dict[str, Any]:
        """Load metadata from HDF5
        
        Args:
            key: Metadata key/identifier
            
        Returns:
            Dictionary of metadata
            
        Raises:
            KeyError: If metadata key not found
            IOError: If metadata load fails
        """
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                metadata_group = f['metadata']
                
                if key not in metadata_group:
                    raise KeyError(f"Metadata key '{key}' not found")
                
                dataset = metadata_group[key]
                metadata_json = dataset[()]
                
                # Handle bytes vs string
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode('utf-8')
                
                metadata = json.loads(metadata_json)
            
            self.logger.info(f"Loaded metadata: {key}")
            
            return metadata
            
        except KeyError:
            raise
        except Exception as e:
            raise IOError(f"Failed to load metadata '{key}': {e}") from e
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored data
        
        Returns:
            Dictionary with counts and sizes of stored data
        """
        info = {
            'hdf5_file': str(self.hdf5_file),
            'file_size_mb': 0.0,
            'n_ground_states': 0,
            'n_observables': 0,
            'n_latent_representations': 0,
            'n_qvae_checkpoints': 0,
        }
        
        # Get file size
        if self.hdf5_file.exists():
            info['file_size_mb'] = self.hdf5_file.stat().st_size / (1024 * 1024)
        
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                # Count ground states
                if 'ground_states' in f:
                    info['n_ground_states'] = len(f['ground_states'].keys())
                
                # Count observables
                if 'observables' in f and 'data' in f['observables']:
                    info['n_observables'] = f['observables']['data'].attrs.get('n_rows', 0)
                
                # Count latent representations
                if 'latent_representations' in f:
                    info['n_latent_representations'] = len(f['latent_representations'].keys())
        except Exception as e:
            self.logger.warning(f"Error reading storage info: {e}")
        
        # Count Q-VAE checkpoints
        checkpoint_dir = self.data_dir / "qvae_checkpoints"
        if checkpoint_dir.exists():
            info['n_qvae_checkpoints'] = len(list(checkpoint_dir.glob("qvae_L*.pt")))
        
        return info


    def load_groundstates_from_dmrg_hdf5(
        self,
        filepath: Path
    ) -> Dict[Tuple[float, int], Any]:
        """Load ground states from DMRG-generated HDF5 file.
        
        This loads ground states from the HDF5 format produced by
        generate_groundstates.py (DMRG/ITensor backend).
        
        Supported file formats (both appear in this repo):
        1) Wavefunction format (attrs-based):
           - group attrs: j2_j1, L, chi, timestamp, ...
           - datasets: psi, energy, observables
        2) RDM-feature format (dataset-based):
           - datasets: j2_j1, L, bond_dim, rdm_features, energy, observables, observable_names, ...
           - attrs may be empty
        
        Args:
            filepath: Path to HDF5 file (e.g., groundstates_L6.h5)
            
        Returns:
            Dictionary mapping (j2_j1, L) -> GroundState-like dict with:
            - coefficients: wavefunction array
            - energy: float
            - observables: array of 11 values
            - L: lattice size
            - j2_j1: frustration ratio
            - metadata: dict with chi, timestamp, etc.
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If HDF5 read fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"DMRG HDF5 file not found: {filepath}")
        
        states = {}
        
        try:
            with h5py.File(filepath, 'r') as f:
                for key in sorted(f.keys()):
                    grp = f[key]

                    # j2_j1 and L can be stored either as attrs (wavefunction format)
                    # or as datasets (rdm-feature format).
                    if 'j2_j1' in grp.attrs:
                        j2_j1 = float(grp.attrs['j2_j1'])
                    elif 'j2_j1' in grp:
                        j2_j1 = float(grp['j2_j1'][()])
                    else:
                        raise KeyError("Missing j2_j1 (attr or dataset)")

                    if 'L' in grp.attrs:
                        L = int(grp.attrs['L'])
                    elif 'L' in grp:
                        L = int(grp['L'][()])
                    else:
                        raise KeyError("Missing L (attr or dataset)")

                    # Load coefficients-like vector.
                    # Prefer full wavefunction if present; otherwise fall back to rdm_features.
                    if 'psi' in grp:
                        psi = grp['psi'][:]
                        source = 'dmrg_hdf5_psi'
                    elif 'rdm_features' in grp:
                        psi = grp['rdm_features'][:]
                        source = 'dmrg_hdf5_rdm_features'
                    else:
                        raise KeyError("Missing psi/rdm_features dataset")

                    energy = float(grp['energy'][()])
                    observables = grp['observables'][:]
                    
                    # Build metadata
                    metadata = {
                        'chi': int(grp.attrs.get('chi', grp.get('bond_dim', [0])[()] if 'bond_dim' in grp else 0)),
                        'timestamp': grp.attrs.get('timestamp', 'unknown'),
                        'hilbert_dim': len(psi),
                        'source': source,
                    }
                    
                    # Get observable names if available
                    if 'observable_names' in grp.attrs:
                        metadata['observable_names'] = list(grp.attrs['observable_names'])
                    elif 'observable_names' in grp:
                        try:
                            metadata['observable_names'] = [
                                x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else str(x)
                                for x in grp['observable_names'][:]
                            ]
                        except Exception:
                            # Keep going; downstream uses defaults if names missing.
                            pass
                    
                    states[(j2_j1, L)] = {
                        'coefficients': psi,
                        'energy': energy,
                        'observables': observables,
                        'L': L,
                        'j2_j1': j2_j1,
                        'metadata': metadata
                    }
            
            self.logger.info(
                f"Loaded {len(states)} ground states from DMRG HDF5: {filepath}"
            )
            
            return states
            
        except Exception as e:
            raise IOError(
                f"Failed to load DMRG HDF5 file {filepath}: {e}"
            ) from e
    
    def convert_dmrg_to_internal_format(
        self,
        dmrg_filepath: Path
    ) -> None:
        """Convert DMRG HDF5 file to internal storage format.
        
        Loads ground states from DMRG-generated HDF5 and saves them
        to the internal j1j2_data.h5 format for use with the rest
        of the pipeline.
        
        Args:
            dmrg_filepath: Path to DMRG HDF5 file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If conversion fails
        """
        from src.ed_module import GroundState
        
        # Load from DMRG format
        dmrg_states = self.load_groundstates_from_dmrg_hdf5(dmrg_filepath)
        
        # Convert and save each state
        for (j2_j1, L), state_data in dmrg_states.items():
            coeffs = state_data['coefficients']
            # Ensure complex dtype for GroundState API; rdm_features may be real-valued.
            if not np.iscomplexobj(coeffs):
                coeffs = coeffs.astype(np.float64) + 0.0j
            else:
                coeffs = coeffs.astype(np.complex128)

            # Normalize (some inputs are feature vectors rather than true wavefunctions).
            # If the vector is all zeros, skip with a clear error.
            norm = np.linalg.norm(coeffs)
            if norm == 0.0:
                raise ValueError(
                    f"Cannot convert state with zero-norm coefficients for L={L}, j2_j1={j2_j1}"
                )
            coeffs = coeffs / norm

            # Create GroundState object
            ground_state = GroundState(
                coefficients=coeffs,
                energy=state_data['energy'],
                basis=None,  # Not available from DMRG
                j2_j1=j2_j1,
                L=L,
                metadata=state_data['metadata']
            )
            
            # Save to internal format
            self.save_ground_state(ground_state, j2_j1, L)
        
        self.logger.info(
            f"Converted {len(dmrg_states)} states from DMRG format to internal storage"
        )
    
    def get_precomputed_observables_from_dmrg(
        self,
        dmrg_filepath: Path
    ) -> pd.DataFrame:
        """Extract precomputed observables from DMRG HDF5 file.
        
        The DMRG file contains observables computed during the sweep.
        This extracts them into a DataFrame compatible with the rest
        of the pipeline.
        
        Args:
            dmrg_filepath: Path to DMRG HDF5 file
            
        Returns:
            DataFrame with columns: j2_j1, L, and all observable names
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If extraction fails
        """
        dmrg_states = self.load_groundstates_from_dmrg_hdf5(dmrg_filepath)
        
        # Default observable names
        default_obs_names = [
            'energy', 'energy_density', 'staggered_magnetization',
            'stripe_order', 'plaquette_order', 'S_pi_pi', 'S_pi_0',
            'entanglement_entropy', 'nematic_order', 'dimer_order_x', 'dimer_order_y'
        ]
        
        rows = []
        for (j2_j1, L), state_data in sorted(dmrg_states.items()):
            row = {'j2_j1': j2_j1, 'L': L}
            
            # Get observable names
            obs_names = state_data['metadata'].get('observable_names', default_obs_names)
            observables = state_data['observables']
            
            # Add each observable
            for i, name in enumerate(obs_names):
                if i < len(observables):
                    row[name] = observables[i]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        self.logger.info(
            f"Extracted {len(df)} observable rows from DMRG HDF5"
        )
        
        return df
