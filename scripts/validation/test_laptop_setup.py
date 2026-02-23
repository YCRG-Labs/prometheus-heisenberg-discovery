#!/usr/bin/env python3
"""Quick test to verify laptop setup is working"""

import sys
from pathlib import Path

print("="*70)
print("Testing Laptop Setup")
print("="*70)
print()

# Test 1: Import all required modules
print("Test 1: Importing modules...")
try:
    from src.config import Config
    from src.logging_config import setup_logging
    from src.ed_module import EDModule
    from src.qvae_module import QVAEModule
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load laptop config
print("\nTest 2: Loading laptop configuration...")
try:
    config = Config.from_yaml('configs/laptop_config.yaml')
    print(f"✓ Config loaded")
    print(f"  Lattice sizes: {config.ed_parameters.lattice_sizes}")
    print(f"  j2/j1 range: [{config.ed_parameters.j2_j1_min}, {config.ed_parameters.j2_j1_max}]")
    print(f"  Step: {config.ed_parameters.j2_j1_step}")
    print(f"  Parallel: {config.ed_parameters.parallel}")
    print(f"  n_processes: {config.ed_parameters.n_processes}")
except Exception as e:
    print(f"✗ Config load failed: {e}")
    sys.exit(1)

# Test 3: Setup logging
print("\nTest 3: Setting up logging...")
try:
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file
    )
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Test log message")
    print("✓ Logging configured successfully")
except Exception as e:
    print(f"✗ Logging setup failed: {e}")
    sys.exit(1)

# Test 4: Check directories
print("\nTest 4: Checking directories...")
dirs_to_check = ['data', 'output', 'logs', 'checkpoints']
for dir_name in dirs_to_check:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"✓ {dir_name}/ exists")
    else:
        print(f"  Creating {dir_name}/...")
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_name}/ created")

# Test 5: Calculate expected runtime
print("\nTest 5: Estimating runtime...")
j2_j1_values = config.get_j2_j1_values()
lattice_sizes = config.ed_parameters.lattice_sizes
total_points = len(j2_j1_values) * len(lattice_sizes)
print(f"  Total parameter points: {total_points}")
print(f"  Estimated ED time: {total_points * 2}-{total_points * 4} minutes ({total_points * 2 / 60:.1f}-{total_points * 4 / 60:.1f} hours)")
print(f"  Estimated Q-VAE time: 2-4 hours")
print(f"  Estimated Analysis time: 1-2 hours")
print(f"  Total estimated time: {total_points * 2 / 60 + 3:.1f}-{total_points * 4 / 60 + 6:.1f} hours")

# Test 6: Check system resources
print("\nTest 6: Checking system resources...")
try:
    import psutil
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    disk = psutil.disk_usage('.')
    
    print(f"  CPU cores: {cpu_count}")
    print(f"  RAM: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available")
    print(f"  Disk: {disk.free / 1024**3:.1f} GB free")
    
    if memory.total < 15 * 1024**3:
        print("  ⚠ Warning: Less than 16 GB RAM. Use sequential processing.")
    if disk.free < 50 * 1024**3:
        print("  ⚠ Warning: Less than 50 GB free disk space.")
    
    print("✓ System resources checked")
except Exception as e:
    print(f"  Could not check system resources: {e}")

print()
print("="*70)
print("✓ ALL TESTS PASSED - Ready to run analysis!")
print("="*70)
print()
print("To start the analysis:")
print("  python run_laptop_analysis.py --stage all")
print()
print("To monitor progress:")
print("  python monitor_laptop.py")
print()
print("Expected runtime: 9-16 hours (run overnight recommended)")
print("="*70)
