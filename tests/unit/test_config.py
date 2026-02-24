"""Unit tests for configuration management"""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.config import (
    Config,
    EDParameters,
    QVAEArchitecture,
    TrainingParameters,
    AnalysisParameters,
    PathConfig,
    LoggingConfig
)


class TestEDParameters:
    """Test ED parameter validation"""
    
    def test_valid_lattice_sizes(self):
        """Test valid lattice sizes are accepted (only even L with Sz=0 sector: 4, 6)"""
        params = EDParameters(lattice_sizes=[4, 6])
        assert params.lattice_sizes == [4, 6]
    
    def test_invalid_lattice_size(self):
        """Test invalid lattice size raises error"""
        with pytest.raises(ValueError, match="not in"):
            EDParameters(lattice_sizes=[4, 7])
    
    def test_j2_j1_range_validation(self):
        """Test j2_j1 range validation"""
        with pytest.raises(ValueError, match="must be less than"):
            EDParameters(j2_j1_min=0.7, j2_j1_max=0.3)
    
    def test_j2_j1_bounds(self):
        """Test j2_j1 bounds validation"""
        with pytest.raises(ValueError):
            EDParameters(j2_j1_min=-0.1)
        
        with pytest.raises(ValueError):
            EDParameters(j2_j1_max=1.5)


class TestQVAEArchitecture:
    """Test Q-VAE architecture validation"""
    
    def test_valid_architecture(self):
        """Test valid architecture is accepted"""
        arch = QVAEArchitecture(
            latent_dim=8,
            encoder_layers=[512, 256, 128],
            decoder_layers=[128, 256, 512]
        )
        assert arch.latent_dim == 8
    
    def test_invalid_layer_dimensions(self):
        """Test negative layer dimensions raise error"""
        with pytest.raises(ValueError, match="positive"):
            QVAEArchitecture(encoder_layers=[512, -256, 128])


class TestTrainingParameters:
    """Test training parameter validation"""
    
    def test_valid_parameters(self):
        """Test valid training parameters"""
        params = TrainingParameters(
            learning_rate=0.001,
            batch_size=32,
            max_epochs=1000
        )
        assert params.learning_rate == 0.001
    
    def test_invalid_learning_rate(self):
        """Test negative learning rate raises error"""
        with pytest.raises(ValueError):
            TrainingParameters(learning_rate=-0.001)


class TestAnalysisParameters:
    """Test analysis parameter validation"""
    
    def test_correlation_threshold_bounds(self):
        """Test correlation threshold is bounded [0, 1]"""
        with pytest.raises(ValueError):
            AnalysisParameters(correlation_threshold=1.5)
        
        with pytest.raises(ValueError):
            AnalysisParameters(correlation_threshold=-0.1)


class TestLoggingConfig:
    """Test logging configuration"""
    
    def test_valid_logging_level(self):
        """Test valid logging levels"""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            config = LoggingConfig(level=level)
            assert config.level == level
    
    def test_invalid_logging_level(self):
        """Test invalid logging level raises error"""
        with pytest.raises(ValueError, match="must be one of"):
            LoggingConfig(level='INVALID')


class TestConfig:
    """Test main Config class"""
    
    def test_default_config(self):
        """Test default configuration is valid"""
        config = Config()
        assert config.ed_parameters.lattice_sizes == [4, 5, 6]
        assert config.qvae_architecture.latent_dim == 8
    
    def test_from_yaml(self):
        """Test loading configuration from YAML"""
        config_dict = {
            'ed_parameters': {
                'lattice_sizes': [4, 5],
                'j2_j1_min': 0.4,
                'j2_j1_max': 0.6,
                'j2_j1_step': 0.02
            },
            'qvae_architecture': {
                'latent_dim': 10
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = Config.from_yaml(temp_path)
            assert config.ed_parameters.lattice_sizes == [4, 5]
            assert config.ed_parameters.j2_j1_min == 0.4
            assert config.qvae_architecture.latent_dim == 10
        finally:
            Path(temp_path).unlink()
    
    def test_from_yaml_nonexistent_file(self):
        """Test loading from nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml('nonexistent.yaml')
    
    def test_to_yaml(self):
        """Test saving configuration to YAML"""
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_yaml(temp_path)
            assert Path(temp_path).exists()
            
            # Load it back and verify
            loaded_config = Config.from_yaml(temp_path)
            assert loaded_config.ed_parameters.lattice_sizes == config.ed_parameters.lattice_sizes
        finally:
            Path(temp_path).unlink()
    
    def test_get_j2_j1_values(self):
        """Test getting j2_j1 values for parameter sweep"""
        config = Config()
        config.ed_parameters.j2_j1_min = 0.3
        config.ed_parameters.j2_j1_max = 0.35
        config.ed_parameters.j2_j1_step = 0.01
        
        values = config.get_j2_j1_values()
        assert len(values) == 6  # 0.30, 0.31, 0.32, 0.33, 0.34, 0.35
        assert abs(values[0] - 0.3) < 1e-10
        assert abs(values[-1] - 0.35) < 1e-10
    
    def test_get_parameter_points(self):
        """Test getting all (j2_j1, L) parameter points"""
        config = Config()
        config.ed_parameters.lattice_sizes = [4, 5]
        config.ed_parameters.j2_j1_min = 0.3
        config.ed_parameters.j2_j1_max = 0.32
        config.ed_parameters.j2_j1_step = 0.01
        
        points = config.get_parameter_points()
        # Should have 3 j2_j1 values * 2 lattice sizes = 6 points
        assert len(points) == 6
        assert (0.3, 4) in points
        assert (0.32, 5) in points
    
    def test_validate_creates_directories(self):
        """Test validate creates necessary directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.paths.data_dir = str(Path(tmpdir) / "data")
            config.paths.output_dir = str(Path(tmpdir) / "output")
            config.paths.checkpoint_dir = str(Path(tmpdir) / "checkpoints")
            config.logging.file = str(Path(tmpdir) / "logs" / "test.log")
            
            config.validate()
            
            assert Path(config.paths.data_dir).exists()
            assert Path(config.paths.output_dir).exists()
            assert Path(config.paths.checkpoint_dir).exists()
            assert Path(config.logging.file).parent.exists()
