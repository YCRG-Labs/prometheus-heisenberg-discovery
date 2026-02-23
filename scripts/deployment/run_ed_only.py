#!/usr/bin/env python3
"""Run ED computation only without loading CUDA/GPU libraries

This script runs only the ED parameter sweep without initializing
GPU libraries, reducing memory footprint for parallel processing.
"""

import os
# Disable CUDA before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from src.config import Config
from src.logging_config import setup_logging
from src.ed_module import EDModule


def main():
    parser = argparse.ArgumentParser(description='Run ED sweep only')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--n-processes', type=int, default=2,
                       help='Number of parallel processes (default: 2)')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    config.ed_parameters.n_processes = args.n_processes
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running ED sweep with {args.n_processes} processes")
    logger.info("GPU disabled to reduce memory footprint")
    
    # Run ED sweep
    ed_module = EDModule(config)
    states = ed_module.run_parameter_sweep(
        parallel=True,
        n_processes=args.n_processes,
        resume=True
    )
    
    logger.info(f"ED sweep complete: {len(states)} states computed")
    logger.info("Checkpoints saved. You can now run Q-VAE training separately.")
    
    return 0


if __name__ == "__main__":
    exit(main())
