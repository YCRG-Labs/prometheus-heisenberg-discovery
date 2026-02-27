#!/usr/bin/env python3
"""Convert DMRG data and run full analysis pipeline"""

import sys
from pathlib import Path
from src.data_storage import DataStorage
from src.config import Config

print("=" * 60)
print("Converting DMRG data to internal format")
print("=" * 60)

# Load config
config = Config.from_yaml('configs/vm_config.yaml')
storage = DataStorage(config)

# Convert L=6 data
l6_file = Path('data/groundstates_L6_rdm.h5')
if l6_file.exists():
    print(f"\nConverting {l6_file}...")
    storage.convert_dmrg_to_internal_format(l6_file)
    print("✓ L=6 conversion complete")
else:
    print(f"⚠ Warning: {l6_file} not found")

# Convert L=8 data
l8_file = Path('data/groundstates_L8_rdm.h5')
if l8_file.exists():
    print(f"\nConverting {l8_file}...")
    storage.convert_dmrg_to_internal_format(l8_file)
    print("✓ L=8 conversion complete")
else:
    print(f"⚠ Warning: {l8_file} not found")

# Also extract and save observables
print("\n" + "=" * 60)
print("Extracting observables from DMRG data")
print("=" * 60)

import pandas as pd

all_obs = []

if l6_file.exists():
    print(f"\nExtracting observables from {l6_file}...")
    obs_l6 = storage.get_precomputed_observables_from_dmrg(l6_file)
    all_obs.append(obs_l6)
    print(f"✓ Extracted {len(obs_l6)} rows for L=6")

if l8_file.exists():
    print(f"\nExtracting observables from {l8_file}...")
    obs_l8 = storage.get_precomputed_observables_from_dmrg(l8_file)
    all_obs.append(obs_l8)
    print(f"✓ Extracted {len(obs_l8)} rows for L=8")

if all_obs:
    combined_obs = pd.concat(all_obs, ignore_index=True)
    storage.save_observables(combined_obs)
    print(f"\n✓ Saved {len(combined_obs)} total observable rows")

print("\n" + "=" * 60)
print("Conversion complete! Now run:")
print("  python main_pipeline.py --config configs/vm_config.yaml --skip-ed --skip-training")
print("=" * 60)
