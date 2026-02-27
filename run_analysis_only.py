#!/usr/bin/env python3
"""
Run complete analysis pipeline without visualizations.
Generates all numerical results needed for the paper.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

from src.config import Config
from src.data_storage import DataStorage
from src.qvae_module import QVAEModule
from src.order_parameter_discovery import OrderParameterDiscovery
from src.critical_point_detection import CriticalPointDetection
from src.finite_size_scaling import FiniteSizeScaling
from src.logging_config import setup_logging

def main():
    setup_logging(level='INFO')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("J1-J2 ANALYSIS PIPELINE (DATA ONLY)")
    logger.info("=" * 80)
    
    # Load config
    config = Config.from_yaml('configs/vm_config.yaml')
    storage = DataStorage(config)
    
    # Step 1: Convert DMRG data if needed
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading/Converting DMRG Data")
    logger.info("=" * 80)
    
    all_obs = []
    states = {}
    
    for L in [6, 8]:
        dmrg_file = Path(f'data/groundstates_L{L}_rdm.h5')
        
        if not dmrg_file.exists():
            logger.warning(f"DMRG file not found: {dmrg_file}")
            continue
        
        logger.info(f"\nProcessing L={L}...")
        
        # Convert to internal format
        storage.convert_dmrg_to_internal_format(dmrg_file)
        
        # Extract observables
        obs = storage.get_precomputed_observables_from_dmrg(dmrg_file)
        all_obs.append(obs)
        
        # Load ground states
        states_L = storage.load_ground_states_for_lattice_size(L)
        states.update(states_L)
        
        logger.info(f"✓ Loaded {len(states_L)} ground states for L={L}")
    
    # Save combined observables
    if all_obs:
        observables = pd.concat(all_obs, ignore_index=True)
        storage.save_observables(observables)
        logger.info(f"\n✓ Saved {len(observables)} observable rows")
    else:
        logger.error("No observables loaded!")
        sys.exit(1)
    
    # Step 2: Load Q-VAE models and encode
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Loading Q-VAE Models and Encoding")
    logger.info("=" * 80)
    
    qvae_module = QVAEModule(config)
    
    for L in [6, 8]:
        try:
            qvae_module.load_model(L, storage)
            logger.info(f"✓ Loaded Q-VAE model for L={L}")
        except Exception as e:
            logger.error(f"Failed to load model for L={L}: {e}")
            sys.exit(1)
    
    # Encode all states to latent space
    logger.info("\nEncoding ground states to latent space...")
    latent_reps = qvae_module.encode_all(states)
    storage.save_latent_representations(latent_reps)
    logger.info(f"✓ Encoded {len(latent_reps)} states")
    
    # Step 3: Order parameter discovery
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Order Parameter Discovery")
    logger.info("=" * 80)
    
    opd_module = OrderParameterDiscovery(config)
    discovery_results = opd_module.discover_order_parameters(latent_reps, observables)
    
    storage.save_metadata('order_parameter_discovery', discovery_results)
    
    logger.info("\nDiscovered order parameters:")
    for latent_dim, observable in discovery_results.get('discovered_order_parameters', {}).items():
        logger.info(f"  {latent_dim} <-> {observable}")
    
    # Step 4: Critical point detection
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Critical Point Detection")
    logger.info("=" * 80)
    
    cpd_module = CriticalPointDetection(config, qvae_module)
    detections = cpd_module.detect_all_methods(states, latent_reps)
    
    if len(detections) > 0:
        j2_c_ensemble, unc_ensemble = cpd_module.ensemble_estimate(detections)
        detections['ensemble'] = (j2_c_ensemble, unc_ensemble)
        logger.info(f"\n✓ Ensemble critical point: J2/J1_c = {j2_c_ensemble:.4f} ± {unc_ensemble:.4f}")
    
    storage.save_metadata('critical_point_detection', {
        'detections': {k: {'j2_j1_c': v[0], 'uncertainty': v[1]} 
                      for k, v in detections.items()}
    })
    
    # Step 5: Finite-size scaling
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Finite-Size Scaling Analysis")
    logger.info("=" * 80)
    
    fss_module = FiniteSizeScaling(config)
    
    # Get initial critical point estimate
    if 'ensemble' in detections:
        j2_c_init, _ = detections['ensemble']
    else:
        j2_c_init = 0.5
    
    scaling_results = {}
    key_observables = ['staggered_magnetization', 'S_pi_pi', 'energy_density']
    
    # Convert to wide format if needed
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
            logger.warning(f"Observable {obs_name} not found")
            continue
        
        logger.info(f"\nScaling analysis for {obs_name}...")
        
        j2_j1 = obs_wide['j2_j1'].values
        L = obs_wide['L'].values
        obs_values = obs_wide[obs_name].values
        
        # Remove NaN
        mask = ~np.isnan(obs_values)
        j2_j1 = j2_j1[mask]
        L = L[mask]
        obs_values = obs_values[mask]
        
        if len(j2_j1) < 10:
            logger.warning(f"Insufficient data for {obs_name}")
            continue
        
        try:
            result = fss_module.optimize_collapse(j2_j1, L, obs_values, j2_c_init)
            bootstrap_result = fss_module.bootstrap_exponents(
                j2_j1, L, obs_values, j2_c_init, n_bootstrap=100
            )
            
            result['j2_j1_c_uncertainty'] = bootstrap_result['j2_j1_c'][1]
            result['nu_uncertainty'] = bootstrap_result['nu'][1]
            result['x_O_uncertainty'] = bootstrap_result['x_O'][1]
            
            scaling_results[obs_name] = result
            
            logger.info(f"  J2/J1_c = {result['j2_j1_c']:.4f} ± {result['j2_j1_c_uncertainty']:.4f}")
            logger.info(f"  ν = {result['nu']:.4f} ± {result['nu_uncertainty']:.4f}")
            logger.info(f"  x_O = {result['x_O']:.4f} ± {result['x_O_uncertainty']:.4f}")
            
        except Exception as e:
            logger.error(f"Scaling analysis failed for {obs_name}: {e}")
    
    storage.save_metadata('finite_size_scaling', scaling_results)
    
    # Step 6: Save comprehensive summary
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Saving Results Summary")
    logger.info("=" * 80)
    
    summary = {
        'configuration': config.model_dump(),
        'order_parameters': {
            'discovered': discovery_results.get('discovered_order_parameters', {}),
            'correlations': discovery_results.get('correlation_matrix', {}),
            'validation': discovery_results.get('validation_results', {})
        },
        'critical_points': {
            method: {'j2_j1_c': j2_c, 'uncertainty': unc}
            for method, (j2_c, unc) in detections.items()
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
    
    storage.save_metadata('analysis_summary', summary)
    
    # Also save as JSON for easy access
    import json
    summary_file = Path('results/analysis_summary.json')
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"✓ Saved summary to {summary_file}")
    
    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE - SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\nOutput files:")
    logger.info(f"  • results/output/observables.csv - All observable values")
    logger.info(f"  • results/analysis_summary.json - Complete analysis results")
    logger.info(f"  • data/j1j2_data.h5 - Ground states and latent representations")
    
    logger.info("\nKey results:")
    if 'ensemble' in detections:
        j2_c, unc = detections['ensemble']
        logger.info(f"  • Critical point: J2/J1 = {j2_c:.4f} ± {unc:.4f}")
    
    logger.info(f"  • Discovered {len(discovery_results.get('discovered_order_parameters', {}))} order parameters")
    logger.info(f"  • Finite-size scaling for {len(scaling_results)} observables")
    
    logger.info("\n" + "=" * 80)

if __name__ == '__main__':
    main()
