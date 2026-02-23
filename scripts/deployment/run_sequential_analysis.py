#!/usr/bin/env python3
"""Sequential Analysis Script - Process one lattice size at a time

This script runs the full analysis but processes each lattice size sequentially
to avoid memory issues with parallel processing on systems with limited virtual memory.
"""

import argparse
import logging
from pathlib import Path
import gc
import torch

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


def run_for_single_lattice_size(config: Config, L: int, logger: logging.Logger):
    """Run complete analysis for a single lattice size"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Lattice Size L={L}")
    logger.info(f"{'='*80}\n")
    
    # Create temporary config with only this lattice size
    temp_config = Config.from_dict(config.to_dict())
    temp_config.ed_parameters.lattice_sizes = [L]
    temp_config.ed_parameters.n_processes = 1  # Force sequential for ED
    
    # Step 1: ED computation
    logger.info(f"Step 1: Computing ground states for L={L}")
    ed_module = EDModule(temp_config)
    states = ed_module.run_parameter_sweep(parallel=False, resume=True)
    logger.info(f"Computed {len(states)} ground states")
    
    # Step 2: Compute observables
    logger.info(f"Step 2: Computing observables for L={L}")
    obs_module = ObservableModule(temp_config)
    observables_df = obs_module.compute_for_sweep(states)
    logger.info(f"Computed observables: {observables_df.shape}")
    
    # Save intermediate results
    storage = DataStorage(temp_config)
    storage.save_observables(observables_df)
    
    # Clear memory
    del ed_module
    del states
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"Completed L={L}\n")
    return observables_df


def main():
    parser = argparse.ArgumentParser(
        description='Sequential J1-J2 Analysis (one lattice size at a time)'
    )
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
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
    
    logger.info("="*80)
    logger.info("J1-J2 Heisenberg Sequential Analysis Pipeline")
    logger.info("="*80)
    
    # Get all lattice sizes
    lattice_sizes = config.ed_parameters.lattice_sizes
    logger.info(f"Will process lattice sizes: {lattice_sizes}")
    
    # Process each lattice size sequentially
    all_observables = []
    for L in lattice_sizes:
        try:
            obs_df = run_for_single_lattice_size(config, L, logger)
            all_observables.append(obs_df)
        except Exception as e:
            logger.error(f"Failed to process L={L}: {e}")
            logger.exception("Full traceback:")
            continue
    
    if not all_observables:
        logger.error("No lattice sizes completed successfully")
        return 1
    
    # Combine results
    logger.info("\n" + "="*80)
    logger.info("Combining results from all lattice sizes")
    logger.info("="*80)
    
    import pandas as pd
    combined_observables = pd.concat(all_observables, ignore_index=True)
    
    # Save combined results
    storage = DataStorage(config)
    storage.save_observables(combined_observables)
    
    logger.info(f"\nTotal observables computed: {combined_observables.shape}")
    logger.info("Sequential analysis complete!")
    logger.info("\nNext steps:")
    logger.info("1. Train Q-VAE: python -c 'from src.qvae_module import QVAEModule; ...'")
    logger.info("2. Run analysis modules separately")
    logger.info("3. Generate visualizations")
    
    return 0


if __name__ == "__main__":
    exit(main())
