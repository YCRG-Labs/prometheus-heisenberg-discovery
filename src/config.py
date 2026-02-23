"""Configuration management for J1-J2 Heisenberg Prometheus Framework"""

from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class EDParameters(BaseModel):
    """Exact diagonalization parameters"""
    lattice_sizes: List[int] = Field(default=[4, 5, 6])
    j2_j1_min: float = Field(default=0.3, ge=0.0, le=1.0)
    j2_j1_max: float = Field(default=0.7, ge=0.0, le=1.0)
    j2_j1_step: float = Field(default=0.01, gt=0.0)
    lanczos_tol: float = Field(default=1e-10, gt=0.0)
    use_translation_symmetry: bool = Field(default=True)
    # Parallel processing configuration
    parallel: bool = Field(default=True)
    n_processes: Optional[int] = Field(default=None, ge=1)
    # Memory optimization configuration
    monitor_memory: bool = Field(default=True)
    clear_cache_after_computation: bool = Field(default=True)

    @field_validator('lattice_sizes')
    @classmethod
    def validate_lattice_sizes(cls, v):
        """Validate lattice sizes are in acceptable range"""
        valid_sizes = {4, 5, 6}
        for size in v:
            if size not in valid_sizes:
                raise ValueError(f"Lattice size {size} not in {valid_sizes}")
        return v

    @model_validator(mode='after')
    def validate_j2_j1_range(self):
        """Validate j2_j1 range is consistent"""
        if self.j2_j1_min >= self.j2_j1_max:
            raise ValueError(
                f"j2_j1_min ({self.j2_j1_min}) must be less than "
                f"j2_j1_max ({self.j2_j1_max})"
            )
        return self


class QVAEArchitecture(BaseModel):
    """Q-VAE architecture parameters"""
    latent_dim: int = Field(default=8, gt=0)
    encoder_layers: List[int] = Field(default=[512, 256, 128])
    decoder_layers: List[int] = Field(default=[128, 256, 512])

    @field_validator('encoder_layers', 'decoder_layers')
    @classmethod
    def validate_layers(cls, v):
        """Validate layer dimensions are positive"""
        if not all(dim > 0 for dim in v):
            raise ValueError("All layer dimensions must be positive")
        return v


class TrainingParameters(BaseModel):
    """Q-VAE training parameters"""
    learning_rate: float = Field(default=0.001, gt=0.0)
    batch_size: int = Field(default=32, gt=0)
    max_epochs: int = Field(default=1000, gt=0)
    patience: int = Field(default=50, gt=0)
    beta: float = Field(default=0.1, ge=0.0)
    gradient_clip: float = Field(default=1.0, gt=0.0)
    n_random_seeds: int = Field(default=3, gt=0)
    # GPU configuration
    use_gpu: bool = Field(default=True)
    gpu_id: Optional[int] = Field(default=None, ge=0)


class AnalysisParameters(BaseModel):
    """Analysis parameters"""
    bootstrap_samples: int = Field(default=1000, gt=0)
    correlation_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    significance_level: float = Field(default=0.01, gt=0.0, lt=1.0)


class PathConfig(BaseModel):
    """Path configuration"""
    data_dir: str = Field(default="./data")
    output_dir: str = Field(default="./output")
    checkpoint_dir: str = Field(default="./checkpoints")

    def get_data_path(self) -> Path:
        """Get data directory as Path object"""
        return Path(self.data_dir)

    def get_output_path(self) -> Path:
        """Get output directory as Path object"""
        return Path(self.output_dir)

    def get_checkpoint_path(self) -> Path:
        """Get checkpoint directory as Path object"""
        return Path(self.checkpoint_dir)


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file: Optional[str] = Field(default="./logs/j1j2_prometheus.log")

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        """Validate logging level"""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Logging level must be one of {valid_levels}")
        return v.upper()


class Config(BaseModel):
    """Main configuration class for J1-J2 Heisenberg Prometheus Framework"""
    ed_parameters: EDParameters = Field(default_factory=EDParameters)
    qvae_architecture: QVAEArchitecture = Field(default_factory=QVAEArchitecture)
    training: TrainingParameters = Field(default_factory=TrainingParameters)
    analysis: AnalysisParameters = Field(default_factory=AnalysisParameters)
    paths: PathConfig = Field(default_factory=PathConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config object with loaded parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file
        
        Args:
            path: Path to save YAML configuration file
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def validate(self) -> None:
        """Validate all parameters are in acceptable ranges
        
        This method performs additional validation beyond Pydantic's
        built-in validation. It's called automatically during initialization
        but can be called explicitly for runtime validation.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Pydantic already validates during initialization
        # This method is here for explicit validation calls
        # and can be extended with additional checks
        
        # Validate paths can be created
        for path_name in ['data_dir', 'output_dir', 'checkpoint_dir']:
            path = Path(getattr(self.paths, path_name))
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Cannot create directory {path}: {e}"
                )
        
        # Validate logging file path
        if self.logging.file:
            log_path = Path(self.logging.file)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Cannot create log directory {log_path.parent}: {e}"
                )

    def get_j2_j1_values(self) -> List[float]:
        """Get list of j2_j1 values for parameter sweep
        
        Returns:
            List of j2_j1 values from min to max with specified step
        """
        import numpy as np
        return np.arange(
            self.ed_parameters.j2_j1_min,
            self.ed_parameters.j2_j1_max + self.ed_parameters.j2_j1_step / 2,
            self.ed_parameters.j2_j1_step
        ).tolist()

    def get_parameter_points(self) -> List[tuple]:
        """Get list of all (j2_j1, L) parameter points
        
        Returns:
            List of (j2_j1, L) tuples for full parameter sweep
        """
        j2_j1_values = self.get_j2_j1_values()
        return [
            (j2_j1, L)
            for j2_j1 in j2_j1_values
            for L in self.ed_parameters.lattice_sizes
        ]

    def get_device(self):
        """Get PyTorch device based on configuration
        
        Returns:
            torch.device object (cuda or cpu)
        """
        import torch
        
        if not self.training.use_gpu or not torch.cuda.is_available():
            return torch.device('cpu')
        
        if self.training.gpu_id is not None:
            return torch.device(f'cuda:{self.training.gpu_id}')
        else:
            return torch.device('cuda')  # Use default GPU
