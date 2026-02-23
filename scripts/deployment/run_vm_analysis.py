#!/usr/bin/env python3
"""
VM Analysis Runner - Staged execution for better control and monitoring

This script allows running the analysis in stages:
1. ED computation (most time-consuming)
2. Q-VAE training
3. Analysis and visualization

Usage:
    python run_vm_analysis.py --stage all
    python run_vm_analysis.py --stage ed
    python run_vm_analysis.py --stage qvae
    python run_vm_analysis.py --stage analysis
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import json

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
from src.progress_monitor import log_system_info


class StageTracker:
    """Track completion of analysis stages"""
    
    def __init__(self, checkpoint_file: str = "analysis_progress.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.stages = self.load()
    
    def load(self):
        """Load stage completion status"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'ed': {'completed': False, 'timestamp': None, 'duration': None},
            'qvae': {'completed': False, 'timestamp': None, 'duration': None},
            'analysis': {'completed': False, 'timestamp': None, 'duration': None}
        }
    
    def save(self):
        """Save stage completion status"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.stages, f, indent=2)
    
    def mark_complete(self, stage: str, duration: float):
        """Mark a stage as complete"""
        self.stages[stage] = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'duration': duration
        }
        self.save()
    
    def is_complete(self, stage: str) -> bool:
        """Check if a stage is complete"""
        return self.stages.get(stage, {}).get('completed', False)
    
    def print_status(self):
        """Print current status"""
        print("\n" + "="*60)
        print("Analysis Progress Status")
        print("="*60)
        for stage, info in self.stages.items():
            status = "✓ COMPLETE" if info['completed'] else "✗ PENDING"
            print(f"{stage.upper():12} {status}")
            if info['completed']:
                duration = info.get('duration', 0)
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                print(f"             Duration: {hours}h {minutes}m")
                print(f"             Completed: {info['timestamp']}")
        print("="*60 + "\n")


def run_ed_stage(config: Config, logger: logging.Logger, tracker: StageTracker):
    """Stage 1: Exact Diagonalization computation"""
    
    if tracker.is_complete('ed'):
        logger.info("ED stage already complete. Skipping.")
        return
    
    logger.info("="*80)
    logger.info("STAGE 1: EXACT DIAGONALIZATION")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Log system info
    log_system_info(logger)
    
    # Initialize ED module
    logger.info("Initializing ED module...")
    ed_module = EDModule(config)
    
    # Run parameter sweep
    logger.info("Starting ED parameter sweep...")
    logger.info(f"Lattice sizes: {config.ed_parameters.lattice_sizes}")
    logger.info(f"j2/j1 range: [{config.ed_parameters.j2_j1_min}, {config.ed_parameters.j2_j1_max}]")
    logger.info(f"Step size: {config.ed_parameters.j2_j1_step}")
    
    j2_j1_values = config.get_j2_j1_values()
    total_points = len(j2_j1_values) * len(config.ed_parameters.lattice_sizes)
    logger.info(f"Total parameter points: {total_points}")
    
    states = ed_module.run_parameter_sweep(
        parallel=config.ed_parameters.parallel,
        n_processes=config.ed_parameters.n_processes,
        resume=True
    )
    
    logger.info(f"ED computation complete: {len(states)} states computed")
    
    # Compute observables
    logger.info("Computing observables...")
    obs_module = ObservableModule(config)
    observables_df = obs_module.compute_for_sweep(states)
    
    # Save observables
    storage = DataStorage(config)
    storage.save_observables(observables_df)
    logger.info(f"Observables saved: {observables_df.shape}")
    
    duration = time.time() - start_time
    tracker.mark_complete('ed', duration)
    
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    logger.info(f"ED stage completed in {hours}h {minutes}m")


def run_qvae_stage(config: Config, logger: logging.Logger, tracker: StageTracker):
    """Stage 2: Q-VAE training"""
    
    if not tracker.is_complete('ed'):
        logger.error("ED stage must be completed before Q-VAE training")
        return
    
    if tracker.is_complete('qvae'):
        logger.info("Q-VAE stage already complete. Skipping.")
        return
    
    logger.info("="*80)
    logger.info("STAGE 2: Q-VAE TRAINING")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Log system info
    log_system_info(logger)
    
    # Initialize Q-VAE module
    logger.info("Initializing Q-VAE module...")
    qvae_module = QVAEModule(config)
    
    # Load ground states
    logger.info("Loading ground states...")
    storage = DataStorage(config)
    states = {}
    for L in config.ed_parameters.lattice_sizes:
        L_states = storage.load_ground_states_for_lattice_size(L)
        states.update(L_states)
    logger.info(f"Loaded {len(states)} ground states")
    
    # Train Q-VAE for all lattice sizes
    logger.info("Training Q-VAE models...")
    models = qvae_module.train_all(states)
    logger.info(f"Trained {len(models)} Q-VAE models")
    
    # Encode all states
    logger.info("Encoding states to latent space...")
    latent_representations = qvae_module.encode_all(states, models)
    
    # Save latent representations
    storage.save_latent_representations(latent_representations, "latent_representations.h5")
    logger.info(f"Latent representations saved: {len(latent_representations)} states")
    
    duration = time.time() - start_time
    tracker.mark_complete('qvae', duration)
    
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    logger.info(f"Q-VAE stage completed in {hours}h {minutes}m")


def run_analysis_stage(config: Config, logger: logging.Logger, tracker: StageTracker):
    """Stage 3: Analysis and visualization"""
    
    if not tracker.is_complete('ed') or not tracker.is_complete('qvae'):
        logger.error("ED and Q-VAE stages must be completed before analysis")
        return
    
    if tracker.is_complete('analysis'):
        logger.info("Analysis stage already complete. Skipping.")
        return
    
    logger.info("="*80)
    logger.info("STAGE 3: ANALYSIS AND VISUALIZATION")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Load data
    logger.info("Loading data...")
    storage = DataStorage(config)
    
    observables_df = storage.load_observables("observables_all.csv")
    latent_representations = storage.load_latent_representations("latent_representations.h5")
    logger.info(f"Loaded observables: {observables_df.shape}")
    logger.info(f"Loaded latent representations: {len(latent_representations)} states")
    
    # Order parameter discovery
    logger.info("Discovering order parameters...")
    op_discovery = OrderParameterDiscovery(config)
    order_params = op_discovery.discover_order_parameters(
        latent_representations,
        observables_df
    )
    
    # Save order parameters
    storage.save_analysis_results(order_params, "order_parameters.json")
    logger.info("Order parameters discovered and saved")
    
    # Critical point detection
    logger.info("Detecting critical points...")
    cp_detection = CriticalPointDetection(config)
    critical_points = cp_detection.detect_all_methods(
        latent_representations,
        observables_df
    )
    
    # Save critical points
    storage.save_analysis_results(critical_points, "critical_points.json")
    logger.info("Critical points detected and saved")
    
    # Finite-size scaling
    logger.info("Performing finite-size scaling analysis...")
    fss = FiniteSizeScaling(config)
    scaling_results = fss.analyze_all_observables(observables_df)
    
    # Save scaling results
    storage.save_analysis_results(scaling_results, "scaling_results.json")
    logger.info("Finite-size scaling complete")
    
    # Validation
    logger.info("Validating results...")
    validator = ValidationModule(config)
    validation_results = validator.validate_all(
        latent_representations,
        observables_df,
        order_params,
        critical_points
    )
    
    # Save validation results
    storage.save_analysis_results(validation_results, "validation_results.json")
    logger.info("Validation complete")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = Visualizer(config)
    
    visualizer.plot_phase_diagram(observables_df, order_params)
    visualizer.plot_latent_trajectories(latent_representations)
    visualizer.plot_correlation_heatmap(latent_representations, observables_df)
    visualizer.plot_critical_point_detection(critical_points)
    visualizer.plot_scaling_collapse(scaling_results)
    
    logger.info("All visualizations generated")
    
    # Create summary report
    create_summary_report(config, order_params, critical_points, 
                         scaling_results, validation_results, logger)
    
    duration = time.time() - start_time
    tracker.mark_complete('analysis', duration)
    
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    logger.info(f"Analysis stage completed in {hours}h {minutes}m")


def create_summary_report(config, order_params, critical_points, 
                         scaling_results, validation_results, logger):
    """Create a summary report of key findings"""
    
    output_dir = Path(config.paths.output_dir)
    summary_file = output_dir / "ANALYSIS_SUMMARY.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("J1-J2 HEISENBERG MODEL ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n\n")
        
        # Critical points
        f.write("1. CRITICAL POINTS DETECTED\n\n")
        if 'ensemble_estimate' in critical_points:
            cp = critical_points['ensemble_estimate']
            f.write(f"   j2/j1 critical = {cp['value']:.4f} ± {cp['uncertainty']:.4f}\n\n")
        
        # Order parameters
        f.write("2. ORDER PARAMETERS DISCOVERED\n\n")
        if 'top_correlations' in order_params:
            for i, corr in enumerate(order_params['top_correlations'][:5], 1):
                f.write(f"   {i}. {corr['observable']}: r = {corr['correlation']:.3f}\n")
        f.write("\n")
        
        # Validation
        f.write("3. VALIDATION RESULTS\n\n")
        for key, value in validation_results.items():
            if isinstance(value, dict) and 'passed' in value:
                status = "✓ PASSED" if value['passed'] else "✗ FAILED"
                f.write(f"   {key}: {status}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("For detailed results, see:\n")
        f.write("  - order_parameters.json\n")
        f.write("  - critical_points.json\n")
        f.write("  - scaling_results.json\n")
        f.write("  - validation_results.json\n")
        f.write("  - Phase diagram: phase_diagram.png\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Summary report saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='VM Analysis Runner - Staged execution'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vm_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', 'ed', 'qvae', 'analysis'],
        default='all',
        help='Which stage to run'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file
    )
    logger = logging.getLogger(__name__)
    
    # Initialize stage tracker
    tracker = StageTracker()
    tracker.print_status()
    
    logger.info("="*80)
    logger.info("J1-J2 HEISENBERG VM ANALYSIS")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Stage: {args.stage}")
    
    overall_start = time.time()
    
    try:
        if args.stage in ['all', 'ed']:
            run_ed_stage(config, logger, tracker)
        
        if args.stage in ['all', 'qvae']:
            run_qvae_stage(config, logger, tracker)
        
        if args.stage in ['all', 'analysis']:
            run_analysis_stage(config, logger, tracker)
        
        overall_duration = time.time() - overall_start
        hours = int(overall_duration // 3600)
        minutes = int((overall_duration % 3600) // 60)
        
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {hours}h {minutes}m")
        
        tracker.print_status()
        
        # Create completion marker
        Path("ANALYSIS_COMPLETE").touch()
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
