#!/usr/bin/env python3
"""Main Pipeline for J1-J2 Heisenberg Prometheus Framework

This script orchestrates the complete analysis pipeline:
1. Load configuration
2. Run ED parameter sweep
3. Compute observables
4. Train Q-VAE for all lattice sizes
5. Encode all states to latent space
6. Discover order parameters
7. Detect critical points
8. Perform finite-size scaling
9. Generate all visualizations
10. Save comprehensive results

Usage:
    python main_pipeline.py [--config path/to/config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import time
import numpy as np

# Import all required modules
from src.config import Config
from src.logging_config import setup_logging
from src.ed_module import EDModule
from src.observable_module import ObservableModule
from src.qvae_module import QVAEModule
from src.order_parameter_discovery import OrderParameterDiscovery
from src.critical_point_detection import CriticalPointDetection
from src.finite_size_scaling import FiniteSizeScaling
from src.data_storage import DataStorage
from src.visualization import Visualizer
from src.validation import ValidationModule
from src.progress_monitor import ProgressMonitor, StepTimer, log_system_info


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='J1-J2 Heisenberg Prometheus Analysis Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file (default: configs/default_config.yaml)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoints if available'
    )
    parser.add_argument(
        '--skip-ed',
        action='store_true',
        help='Skip ED computation (load from storage)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip Q-VAE training (load from checkpoints)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def load_configuration(config_path: str) -> Config:
    """Load and validate configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated Config object
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        config = Config.from_yaml(config_path)
        config.validate()
        logger.info("Configuration loaded and validated successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def run_ed_parameter_sweep(
    config: Config,
    ed_module: EDModule,
    storage: DataStorage,
    resume: bool = True
) -> Dict[Tuple[float, int], Any]:
    """Run exact diagonalization parameter sweep
    
    Args:
        config: Configuration object
        ed_module: EDModule instance
        storage: DataStorage instance
        resume: Whether to resume from checkpoints
        
    Returns:
        Dictionary mapping (j2_j1, L) -> GroundState
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 1: Exact Diagonalization Parameter Sweep")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Run parameter sweep
    states = ed_module.run_parameter_sweep(
        parallel=True,
        resume=resume
    )
    
    # Save all ground states to storage
    logger.info("Saving ground states to storage...")
    for (j2_j1, L), state in states.items():
        storage.save_ground_state(state, j2_j1, L)
    
    elapsed_time = time.time() - start_time
    logger.info(f"ED parameter sweep completed in {elapsed_time:.2f} seconds")
    logger.info(f"Computed {len(states)} ground states")
    
    return states


def compute_observables(
    config: Config,
    states: Dict[Tuple[float, int], Any],
    obs_module: ObservableModule,
    storage: DataStorage
) -> Any:
    """Compute physical observables for all states
    
    Args:
        config: Configuration object
        states: Dictionary of ground states
        obs_module: ObservableModule instance
        storage: DataStorage instance
        
    Returns:
        DataFrame with observable values
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 2: Observable Computation")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Compute observables for all states
    observables = obs_module.compute_for_sweep(states)
    
    # Save observables to storage
    logger.info("Saving observables to storage...")
    storage.save_observables(observables)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Observable computation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Computed {len(observables)} observable values")
    
    return observables


def train_qvae_models(
    config: Config,
    states: Dict[Tuple[float, int], Any],
    qvae_module: QVAEModule,
    storage: DataStorage,
    skip_training: bool = False
) -> None:
    """Train Q-VAE models for all lattice sizes
    
    Args:
        config: Configuration object
        states: Dictionary of ground states
        qvae_module: QVAEModule instance
        storage: DataStorage instance
        skip_training: Whether to skip training and load from checkpoints
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 3: Q-VAE Training")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    if skip_training:
        logger.info("Skipping training, loading models from checkpoints...")
        for L in config.ed_parameters.lattice_sizes:
            try:
                qvae_module.load_model(L, storage)
                logger.info(f"Loaded Q-VAE model for L={L}")
            except Exception as e:
                logger.error(f"Failed to load model for L={L}: {e}")
                logger.info(f"Training model for L={L} instead...")
                qvae_module.train_for_lattice_size(states, L)
                qvae_module.save_model(L, storage)
    else:
        # Train models for all lattice sizes
        qvae_module.train_all(states)
        
        # Save all models
        logger.info("Saving Q-VAE models to storage...")
        for L in config.ed_parameters.lattice_sizes:
            qvae_module.save_model(L, storage)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Q-VAE training completed in {elapsed_time:.2f} seconds")


def encode_to_latent_space(
    config: Config,
    states: Dict[Tuple[float, int], Any],
    qvae_module: QVAEModule,
    storage: DataStorage
) -> Dict[Tuple[float, int], Any]:
    """Encode all states to latent representations
    
    Args:
        config: Configuration object
        states: Dictionary of ground states
        qvae_module: QVAEModule instance
        storage: DataStorage instance
        
    Returns:
        Dictionary mapping (j2_j1, L) -> latent vector
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 4: Latent Space Encoding")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Encode all states
    latent_reps = qvae_module.encode_all(states)
    
    # Save latent representations
    logger.info("Saving latent representations to storage...")
    storage.save_latent_representations(latent_reps)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Latent encoding completed in {elapsed_time:.2f} seconds")
    logger.info(f"Encoded {len(latent_reps)} states")
    
    return latent_reps


def discover_order_parameters(
    config: Config,
    latent_reps: Dict[Tuple[float, int], Any],
    observables: Any,
    opd_module: OrderParameterDiscovery,
    storage: DataStorage
) -> Dict[str, Any]:
    """Discover order parameters through correlation analysis
    
    Args:
        config: Configuration object
        latent_reps: Dictionary of latent representations
        observables: DataFrame with observable values
        opd_module: OrderParameterDiscovery instance
        storage: DataStorage instance
        
    Returns:
        Dictionary with discovery results
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 5: Order Parameter Discovery")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Run discovery pipeline
    discovery_results = opd_module.discover_order_parameters(
        latent_reps,
        observables
    )
    
    # Save results
    logger.info("Saving order parameter discovery results...")
    storage.save_metadata('order_parameter_discovery', discovery_results)
    
    # Log discovered order parameters
    logger.info("Discovered order parameters:")
    for latent_dim, observable in discovery_results['discovered_order_parameters'].items():
        logger.info(f"  {latent_dim} <-> {observable}")
    
    # Log validation results
    logger.info("Validation results:")
    for key, value in discovery_results['validation_results'].items():
        logger.info(f"  {key}: {value}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Order parameter discovery completed in {elapsed_time:.2f} seconds")
    
    return discovery_results


def detect_critical_points(
    config: Config,
    states: Dict[Tuple[float, int], Any],
    latent_reps: Dict[Tuple[float, int], Any],
    cpd_module: CriticalPointDetection,
    storage: DataStorage
) -> Dict[str, Any]:
    """Detect critical points using multiple methods
    
    Args:
        config: Configuration object
        states: Dictionary of ground states
        latent_reps: Dictionary of latent representations
        cpd_module: CriticalPointDetection instance
        storage: DataStorage instance
        
    Returns:
        Dictionary with detection results
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 6: Critical Point Detection")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Apply all detection methods
    detections = cpd_module.detect_all_methods(states, latent_reps)
    
    # Compute ensemble estimate
    if len(detections) > 0:
        j2_j1_c_ensemble, uncertainty_ensemble = cpd_module.ensemble_estimate(detections)
        detections['ensemble'] = (j2_j1_c_ensemble, uncertainty_ensemble)
        
        logger.info(f"Ensemble critical point: j2_j1_c = {j2_j1_c_ensemble:.4f} ± {uncertainty_ensemble:.4f}")
    
    # Save results
    logger.info("Saving critical point detection results...")
    storage.save_metadata('critical_point_detection', {
        'detections': {k: {'j2_j1_c': v[0], 'uncertainty': v[1]} 
                      for k, v in detections.items()}
    })
    
    elapsed_time = time.time() - start_time
    logger.info(f"Critical point detection completed in {elapsed_time:.2f} seconds")
    
    return detections


def perform_finite_size_scaling(
    config: Config,
    observables: Any,
    detections: Dict[str, Any],
    fss_module: FiniteSizeScaling,
    storage: DataStorage
) -> Dict[str, Any]:
    """Perform finite-size scaling analysis
    
    Args:
        config: Configuration object
        observables: DataFrame with observable values
        detections: Dictionary with critical point detections
        fss_module: FiniteSizeScaling instance
        storage: DataStorage instance
        
    Returns:
        Dictionary with scaling results
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 7: Finite-Size Scaling Analysis")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Get ensemble critical point estimate
    if 'ensemble' in detections:
        j2_j1_c_init, _ = detections['ensemble']
    elif len(detections) > 0:
        # Use first available detection
        j2_j1_c_init, _ = list(detections.values())[0]
    else:
        logger.warning("No critical point detections available, using midpoint")
        j2_j1_c_init = (config.ed_parameters.j2_j1_min + config.ed_parameters.j2_j1_max) / 2
    
    scaling_results = {}
    
    # Perform scaling analysis for key observables
    key_observables = ['staggered_mag', 'stripe_order', 'energy_density']
    
    # Convert observables to wide format if needed
    if 'observable_name' in observables.columns:
        obs_wide = observables.pivot_table(
            index=['j2_j1', 'L'],
            columns='observable_name',
            values='value'
        ).reset_index()
    else:
        obs_wide = observables
    
    for obs_name in key_observables:
        if obs_name not in obs_wide.columns:
            logger.warning(f"Observable {obs_name} not found, skipping scaling analysis")
            continue
        
        logger.info(f"Performing scaling analysis for {obs_name}...")
        
        # Prepare data
        j2_j1 = obs_wide['j2_j1'].values
        L = obs_wide['L'].values
        obs_values = obs_wide[obs_name].values
        
        # Remove NaN values
        mask = ~np.isnan(obs_values)
        j2_j1 = j2_j1[mask]
        L = L[mask]
        obs_values = obs_values[mask]
        
        if len(j2_j1) < 10:
            logger.warning(f"Insufficient data for {obs_name} scaling analysis")
            continue
        
        try:
            # Optimize scaling collapse
            result = fss_module.optimize_collapse(
                j2_j1, L, obs_values, j2_j1_c_init
            )
            
            # Bootstrap uncertainties
            bootstrap_result = fss_module.bootstrap_exponents(
                j2_j1, L, obs_values, j2_j1_c_init,
                n_bootstrap=100  # Reduced for speed
            )
            
            # Combine results
            result['j2_j1_c_uncertainty'] = bootstrap_result['j2_j1_c'][1]
            result['nu_uncertainty'] = bootstrap_result['nu'][1]
            result['x_O_uncertainty'] = bootstrap_result['x_O'][1]
            
            scaling_results[obs_name] = result
            
            logger.info(f"  j2_j1_c = {result['j2_j1_c']:.4f} ± {result['j2_j1_c_uncertainty']:.4f}")
            logger.info(f"  nu = {result['nu']:.4f} ± {result['nu_uncertainty']:.4f}")
            logger.info(f"  x_O = {result['x_O']:.4f} ± {result['x_O_uncertainty']:.4f}")
            
        except Exception as e:
            logger.error(f"Scaling analysis failed for {obs_name}: {e}")
    
    # Save results
    logger.info("Saving finite-size scaling results...")
    storage.save_metadata('finite_size_scaling', scaling_results)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Finite-size scaling completed in {elapsed_time:.2f} seconds")
    
    return scaling_results


def generate_visualizations(
    config: Config,
    observables: Any,
    latent_reps: Dict[Tuple[float, int], Any],
    discovery_results: Dict[str, Any],
    detections: Dict[str, Any],
    scaling_results: Dict[str, Any],
    qvae_module: QVAEModule,
    visualizer: Visualizer
) -> None:
    """Generate all visualizations
    
    Args:
        config: Configuration object
        observables: DataFrame with observable values
        latent_reps: Dictionary of latent representations
        discovery_results: Order parameter discovery results
        detections: Critical point detection results
        scaling_results: Finite-size scaling results
        qvae_module: QVAEModule instance
        visualizer: Visualizer instance
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 8: Visualization Generation")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # 1. Phase diagram
    logger.info("Generating phase diagram...")
    try:
        visualizer.plot_phase_diagram(observables)
    except Exception as e:
        logger.error(f"Phase diagram generation failed: {e}")
    
    # 2. Latent trajectories
    logger.info("Generating latent trajectory plots...")
    try:
        visualizer.plot_latent_trajectories(latent_reps, projection_method='pca', color_by='j2_j1')
        visualizer.plot_latent_trajectories(latent_reps, projection_method='pca', color_by='L',
                                          save_name='latent_trajectories_by_L.png')
    except Exception as e:
        logger.error(f"Latent trajectory generation failed: {e}")
    
    # 3. Correlation heatmap
    logger.info("Generating correlation heatmap...")
    try:
        if 'correlation_matrix' in discovery_results:
            visualizer.plot_correlation_heatmap(discovery_results['correlation_matrix'])
    except Exception as e:
        logger.error(f"Correlation heatmap generation failed: {e}")
    
    # 4. Critical point detection
    logger.info("Generating critical point detection plots...")
    try:
        # Get detection data for plotting
        from src.critical_point_detection import LatentVarianceMethod
        lv_method = LatentVarianceMethod()
        latent_variance_data = lv_method.compute_latent_variance(latent_reps)
        
        visualizer.plot_critical_point_detection(
            detections,
            latent_variance_data=latent_variance_data
        )
    except Exception as e:
        logger.error(f"Critical point detection plot generation failed: {e}")
    
    # 5. Scaling collapse plots
    logger.info("Generating scaling collapse plots...")
    for obs_name, result in scaling_results.items():
        try:
            # Prepare data
            if 'observable_name' in observables.columns:
                obs_wide = observables.pivot_table(
                    index=['j2_j1', 'L'],
                    columns='observable_name',
                    values='value'
                ).reset_index()
            else:
                obs_wide = observables
            
            j2_j1 = obs_wide['j2_j1'].values
            L = obs_wide['L'].values
            obs_values = obs_wide[obs_name].values
            
            # Remove NaN
            mask = ~np.isnan(obs_values)
            
            visualizer.plot_scaling_collapse(
                result,
                j2_j1[mask], L[mask], obs_values[mask],
                observable_name=obs_name,
                save_name=f'scaling_collapse_{obs_name}.png'
            )
        except Exception as e:
            logger.error(f"Scaling collapse plot for {obs_name} failed: {e}")
    
    # 6. Training curves
    logger.info("Generating training curve plots...")
    for L in config.ed_parameters.lattice_sizes:
        try:
            if L in qvae_module.training_histories:
                visualizer.plot_training_curves(
                    qvae_module.training_histories[L],
                    lattice_size=L,
                    save_name=f'training_curves_L{L}.png'
                )
        except Exception as e:
            logger.error(f"Training curve plot for L={L} failed: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Visualization generation completed in {elapsed_time:.2f} seconds")


def save_comprehensive_results(
    config: Config,
    storage: DataStorage,
    discovery_results: Dict[str, Any],
    detections: Dict[str, Any],
    scaling_results: Dict[str, Any]
) -> None:
    """Save comprehensive analysis results
    
    Args:
        config: Configuration object
        storage: DataStorage instance
        discovery_results: Order parameter discovery results
        detections: Critical point detection results
        scaling_results: Finite-size scaling results
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STEP 9: Saving Comprehensive Results")
    logger.info("=" * 80)
    
    # Create summary report
    summary = {
        'configuration': config.model_dump(),
        'order_parameters': {
            'discovered': discovery_results.get('discovered_order_parameters', {}),
            'validation': discovery_results.get('validation_results', {})
        },
        'critical_points': {
            method: {'j2_j1_c': j2_j1_c, 'uncertainty': unc}
            for method, (j2_j1_c, unc) in detections.items()
        },
        'scaling_exponents': {
            obs: {
                'j2_j1_c': result['j2_j1_c'],
                'nu': result['nu'],
                'x_O': result['x_O'],
                'uncertainties': {
                    'j2_j1_c': result.get('j2_j1_c_uncertainty', 0.0),
                    'nu': result.get('nu_uncertainty', 0.0),
                    'x_O': result.get('x_O_uncertainty', 0.0)
                }
            }
            for obs, result in scaling_results.items()
        }
    }
    
    # Save summary
    storage.save_metadata('analysis_summary', summary)
    
    # Get storage info
    storage_info = storage.get_storage_info()
    logger.info("Storage summary:")
    for key, value in storage_info.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Comprehensive results saved successfully")


def main():
    """Main pipeline execution"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("J1-J2 HEISENBERG PROMETHEUS ANALYSIS PIPELINE")
    logger.info("=" * 80)
    
    # Log system information
    log_system_info()
    
    pipeline_start_time = time.time()
    
    try:
        # Load configuration
        with StepTimer("Configuration Loading"):
            config = load_configuration(args.config)
        
        # Initialize modules
        logger.info("Initializing modules...")
        storage = DataStorage(config)
        ed_module = EDModule(config)
        obs_module = ObservableModule(config)
        qvae_module = QVAEModule(config)
        opd_module = OrderParameterDiscovery(config)
        cpd_module = CriticalPointDetection(config, qvae_module)
        fss_module = FiniteSizeScaling(config)
        visualizer = Visualizer(config)
        
        # Step 1: ED parameter sweep
        with StepTimer("ED Parameter Sweep"):
            if args.skip_ed:
                logger.info("Skipping ED computation, loading from storage...")
                # Load states from storage
                states = {}
                for (j2_j1, L) in config.get_parameter_points():
                    try:
                        state = storage.load_ground_state(j2_j1, L)
                        states[(j2_j1, L)] = state
                    except Exception as e:
                        logger.warning(f"Could not load state for (j2_j1={j2_j1}, L={L}): {e}")
                
                if not states:
                    logger.error("No states loaded from storage. Run without --skip-ed first.")
                    sys.exit(1)
            else:
                states = run_ed_parameter_sweep(config, ed_module, storage, resume=args.resume)
        
        # Step 2: Compute observables
        with StepTimer("Observable Computation"):
            observables = compute_observables(config, states, obs_module, storage)
        
        # Step 3: Train Q-VAE
        with StepTimer("Q-VAE Training"):
            train_qvae_models(config, states, qvae_module, storage, skip_training=args.skip_training)
        
        # Step 4: Encode to latent space
        with StepTimer("Latent Space Encoding"):
            latent_reps = encode_to_latent_space(config, states, qvae_module, storage)
        
        # Step 5: Discover order parameters
        with StepTimer("Order Parameter Discovery"):
            discovery_results = discover_order_parameters(
                config, latent_reps, observables, opd_module, storage
            )
        
        # Step 6: Detect critical points
        with StepTimer("Critical Point Detection"):
            detections = detect_critical_points(
                config, states, latent_reps, cpd_module, storage
            )
        
        # Step 7: Finite-size scaling
        with StepTimer("Finite-Size Scaling"):
            scaling_results = perform_finite_size_scaling(
                config, observables, detections, fss_module, storage
            )
        
        # Step 8: Generate visualizations
        with StepTimer("Visualization Generation"):
            generate_visualizations(
                config, observables, latent_reps, discovery_results,
                detections, scaling_results, qvae_module, visualizer
            )
        
        # Step 9: Save comprehensive results
        with StepTimer("Results Saving"):
            save_comprehensive_results(
                config, storage, discovery_results, detections, scaling_results
            )
        
        # Pipeline complete
        pipeline_elapsed_time = time.time() - pipeline_start_time
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {pipeline_elapsed_time:.2f} seconds ({pipeline_elapsed_time/60:.2f} minutes)")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
