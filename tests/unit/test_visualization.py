"""
Unit tests for visualization module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')

from src.visualization import Visualizer


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config(temp_output_dir):
    """Create a mock configuration object"""
    config = MagicMock()
    config.output_dir = str(temp_output_dir)
    return config


@pytest.fixture
def visualizer(mock_config):
    """Create a Visualizer instance"""
    return Visualizer(mock_config)


@pytest.fixture
def sample_observables():
    """Create sample observable data"""
    data = []
    for L in [4, 5, 6]:
        for j2_j1 in np.linspace(0.3, 0.7, 10):
            data.append({
                'j2_j1': j2_j1,
                'L': L,
                'observable_name': 'staggered_mag',
                'value': 0.5 * np.exp(-(j2_j1 - 0.5)**2 / 0.1)
            })
            data.append({
                'j2_j1': j2_j1,
                'L': L,
                'observable_name': 'stripe_order',
                'value': 0.3 * np.exp(-(j2_j1 - 0.6)**2 / 0.1)
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_latent_reps():
    """Create sample latent representations"""
    latent_reps = {}
    for L in [4, 5, 6]:
        for j2_j1 in np.linspace(0.3, 0.7, 10):
            # Create 8-dimensional latent vector
            z = np.random.randn(8) + j2_j1  # Correlate with j2_j1
            latent_reps[(j2_j1, L)] = z
    return latent_reps


@pytest.fixture
def sample_correlation_matrix():
    """Create sample correlation matrix"""
    latent_dims = [f'z_{i}' for i in range(8)]
    observables = ['staggered_mag', 'stripe_order', 'energy', 'plaquette_order']
    
    # Create random correlations
    data = np.random.uniform(-1, 1, size=(len(latent_dims), len(observables)))
    
    return pd.DataFrame(data, index=latent_dims, columns=observables)


@pytest.fixture
def sample_detection_results():
    """Create sample critical point detection results"""
    return {
        'latent_variance': (0.48, 0.02),
        'reconstruction_error': (0.50, 0.03),
        'fidelity_susceptibility': (0.49, 0.025),
        'ensemble': (0.49, 0.015)
    }


@pytest.fixture
def sample_scaling_results():
    """Create sample scaling analysis results"""
    return {
        'j2_j1_c': 0.49,
        'nu': 1.0,
        'x_O': 0.125,
        'chi_squared': 0.001,
        'j2_j1_c_uncertainty': 0.02,
        'nu_uncertainty': 0.1,
        'x_O_uncertainty': 0.05
    }


@pytest.fixture
def sample_training_history():
    """Create sample training history"""
    n_epochs = 50
    return {
        'train_loss': list(np.linspace(1.0, 0.1, n_epochs)),
        'val_loss': list(np.linspace(1.1, 0.15, n_epochs)),
        'train_fidelity_loss': list(np.linspace(0.8, 0.05, n_epochs)),
        'val_fidelity_loss': list(np.linspace(0.85, 0.08, n_epochs)),
        'train_kl_loss': list(np.linspace(0.2, 0.05, n_epochs)),
        'val_kl_loss': list(np.linspace(0.25, 0.07, n_epochs))
    }


class TestVisualizerInitialization:
    """Test Visualizer initialization"""
    
    def test_init_creates_output_directories(self, visualizer, temp_output_dir):
        """Test that initialization creates all required output directories"""
        assert visualizer.output_dir.exists()
        assert visualizer.phase_diagram_dir.exists()
        assert visualizer.latent_dir.exists()
        assert visualizer.correlation_dir.exists()
        assert visualizer.critical_point_dir.exists()
        assert visualizer.scaling_dir.exists()
        assert visualizer.training_dir.exists()


class TestPhaseDiagram:
    """Test phase diagram plotting"""
    
    def test_plot_phase_diagram_creates_file(self, visualizer, sample_observables):
        """Test that plot_phase_diagram creates output file"""
        visualizer.plot_phase_diagram(sample_observables)
        
        output_file = visualizer.phase_diagram_dir / 'phase_diagram.png'
        assert output_file.exists()
    
    def test_plot_phase_diagram_with_custom_observables(self, visualizer, sample_observables):
        """Test plotting specific observables"""
        visualizer.plot_phase_diagram(
            sample_observables,
            observable_names=['staggered_mag'],
            save_name='custom_phase_diagram.png'
        )
        
        output_file = visualizer.phase_diagram_dir / 'custom_phase_diagram.png'
        assert output_file.exists()
    
    def test_plot_phase_diagram_with_empty_data(self, visualizer):
        """Test handling of empty observable data"""
        empty_df = pd.DataFrame()
        # Should not raise an error
        visualizer.plot_phase_diagram(empty_df)


class TestLatentTrajectories:
    """Test latent trajectory plotting"""
    
    def test_plot_latent_trajectories_pca(self, visualizer, sample_latent_reps):
        """Test latent trajectory plotting with PCA"""
        visualizer.plot_latent_trajectories(
            sample_latent_reps,
            projection_method='pca',
            color_by='j2_j1'
        )
        
        output_file = visualizer.latent_dir / 'latent_trajectories.png'
        assert output_file.exists()
    
    def test_plot_latent_trajectories_color_by_L(self, visualizer, sample_latent_reps):
        """Test coloring by lattice size"""
        visualizer.plot_latent_trajectories(
            sample_latent_reps,
            projection_method='pca',
            color_by='L',
            save_name='latent_by_L.png'
        )
        
        output_file = visualizer.latent_dir / 'latent_by_L.png'
        assert output_file.exists()
    
    def test_plot_latent_trajectories_empty_data(self, visualizer):
        """Test handling of empty latent representations"""
        empty_dict = {}
        # Should not raise an error
        visualizer.plot_latent_trajectories(empty_dict)


class TestCorrelationHeatmap:
    """Test correlation heatmap plotting"""
    
    def test_plot_correlation_heatmap_creates_file(self, visualizer, sample_correlation_matrix):
        """Test that correlation heatmap creates output file"""
        visualizer.plot_correlation_heatmap(sample_correlation_matrix)
        
        output_file = visualizer.correlation_dir / 'correlation_heatmap.png'
        assert output_file.exists()
    
    def test_plot_correlation_heatmap_custom_range(self, visualizer, sample_correlation_matrix):
        """Test custom color range"""
        visualizer.plot_correlation_heatmap(
            sample_correlation_matrix,
            vmin=-0.5,
            vmax=0.5,
            save_name='custom_heatmap.png'
        )
        
        output_file = visualizer.correlation_dir / 'custom_heatmap.png'
        assert output_file.exists()
    
    def test_plot_correlation_heatmap_empty_data(self, visualizer):
        """Test handling of empty correlation matrix"""
        empty_df = pd.DataFrame()
        # Should not raise an error
        visualizer.plot_correlation_heatmap(empty_df)


class TestCriticalPointDetection:
    """Test critical point detection plotting"""
    
    def test_plot_critical_point_detection_creates_file(self, visualizer, sample_detection_results):
        """Test that critical point detection plot creates output file"""
        # Create sample data for each method
        latent_variance_data = {j2_j1: np.exp(-(j2_j1 - 0.49)**2 / 0.01) 
                               for j2_j1 in np.linspace(0.3, 0.7, 20)}
        
        visualizer.plot_critical_point_detection(
            sample_detection_results,
            latent_variance_data=latent_variance_data
        )
        
        output_file = visualizer.critical_point_dir / 'critical_point_detection.png'
        assert output_file.exists()
    
    def test_plot_critical_point_detection_no_data(self, visualizer, sample_detection_results):
        """Test with no detection data"""
        # Should not raise an error
        visualizer.plot_critical_point_detection(sample_detection_results)


class TestScalingCollapse:
    """Test scaling collapse plotting"""
    
    def test_plot_scaling_collapse_creates_file(self, visualizer, sample_scaling_results):
        """Test that scaling collapse plot creates output file"""
        # Create sample data
        n_points = 30
        j2_j1 = np.random.uniform(0.3, 0.7, n_points)
        L = np.random.choice([4, 5, 6], n_points)
        observable = np.random.randn(n_points) * 0.1 + 0.5
        
        visualizer.plot_scaling_collapse(
            sample_scaling_results,
            j2_j1, L, observable,
            observable_name='Test Observable'
        )
        
        output_file = visualizer.scaling_dir / 'scaling_collapse.png'
        assert output_file.exists()


class TestTrainingCurves:
    """Test training curves plotting"""
    
    def test_plot_training_curves_creates_file(self, visualizer, sample_training_history):
        """Test that training curves plot creates output file"""
        visualizer.plot_training_curves(sample_training_history, lattice_size=4)
        
        output_file = visualizer.training_dir / 'training_curves.png'
        assert output_file.exists()
    
    def test_plot_training_curves_minimal_history(self, visualizer):
        """Test with minimal training history"""
        minimal_history = {
            'train_loss': [1.0, 0.5, 0.2]
        }
        
        visualizer.plot_training_curves(minimal_history)
        
        output_file = visualizer.training_dir / 'training_curves.png'
        assert output_file.exists()
    
    def test_plot_training_curves_empty_history(self, visualizer):
        """Test handling of empty history"""
        empty_history = {}
        # Should not raise an error
        visualizer.plot_training_curves(empty_history)


class TestEnsembleCriticalPoints:
    """Test ensemble critical points plotting"""
    
    def test_plot_ensemble_critical_points_creates_file(self, visualizer, sample_detection_results):
        """Test that ensemble plot creates output file"""
        visualizer.plot_ensemble_critical_points(sample_detection_results)
        
        output_file = visualizer.critical_point_dir / 'ensemble_critical_points.png'
        assert output_file.exists()


class TestSummaryReport:
    """Test summary report generation"""
    
    def test_create_summary_report_creates_file(self, visualizer):
        """Test that summary report creates output file"""
        results = {
            'critical_points': {
                'latent_variance': (0.48, 0.02),
                'ensemble': (0.49, 0.015)
            },
            'order_parameters': {
                'discovered_order_parameters': {
                    'z_0': 'staggered_mag',
                    'z_1': 'stripe_order'
                }
            },
            'scaling': {
                'j2_j1_c': 0.49,
                'nu': 1.0,
                'x_O': 0.125,
                'chi_squared': 0.001
            },
            'validation': {
                'neel_phase_valid': True,
                'stripe_phase_valid': True
            }
        }
        
        visualizer.create_summary_report(results)
        
        output_file = visualizer.output_dir / 'analysis_summary.txt'
        assert output_file.exists()
        
        # Check that file contains expected content
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'CRITICAL POINT DETECTION' in content
            assert 'DISCOVERED ORDER PARAMETERS' in content
            assert 'FINITE-SIZE SCALING ANALYSIS' in content
            assert 'VALIDATION IN KNOWN PHASES' in content
