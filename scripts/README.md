# Scripts Directory

This directory contains utility, validation, and deployment scripts for the J1-J2 Heisenberg Prometheus framework.

## Directory Structure

```
scripts/
├── deployment/          # Deployment and execution scripts
├── utilities/           # Utility and monitoring scripts
├── validation/          # Setup and analysis validation scripts
└── README.md           # This file
```

## Deployment Scripts (`deployment/`)

Scripts for running the analysis pipeline on different platforms.

### Main Execution Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_laptop_analysis.py` | Laptop-optimized analysis | `python scripts/deployment/run_laptop_analysis.py --stage all` |
| `run_vm_analysis.py` | VM/cloud deployment | `python scripts/deployment/run_vm_analysis.py --stage all` |
| `run_sequential_analysis.py` | Memory-constrained execution | `python scripts/deployment/run_sequential_analysis.py` |
| `run_ed_only.py` | ED-only (no GPU libraries) | `python scripts/deployment/run_ed_only.py` |

### Setup Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `vm_setup.sh` | Automated VM setup | `bash scripts/deployment/vm_setup.sh` |
| `prepare_for_vm.sh` | Create deployment package | `bash scripts/deployment/prepare_for_vm.sh` |

### Execution Stages

All Python execution scripts support staged execution:

```bash
# Run all stages
python scripts/deployment/run_laptop_analysis.py --stage all

# Run specific stage
python scripts/deployment/run_laptop_analysis.py --stage ed
python scripts/deployment/run_laptop_analysis.py --stage qvae
python scripts/deployment/run_laptop_analysis.py --stage analysis
```

## Utilities Scripts (`utilities/`)

Helper scripts for monitoring and documentation.

| Script | Purpose | Usage |
|--------|---------|-------|
| `monitor_laptop.py` | Real-time progress monitoring | `python scripts/utilities/monitor_laptop.py` |
| `document_hardware.py` | Hardware documentation | `python scripts/utilities/document_hardware.py` |

### Monitoring

The monitor script provides real-time updates on:
- Current stage progress
- System resource usage (CPU, RAM, disk)
- Estimated time remaining
- Recent log entries

```bash
# Start monitoring
python scripts/utilities/monitor_laptop.py

# Monitor refreshes every 5 seconds
# Press Ctrl+C to exit
```

## Validation Scripts (`validation/`)

Scripts for validating setup and analysis results.

| Script | Purpose | Usage |
|--------|---------|-------|
| `validate_setup.py` | Verify installation | `python scripts/validation/validate_setup.py` |
| `test_laptop_setup.py` | Test laptop configuration | `python scripts/validation/test_laptop_setup.py` |
| `validate_latent_space_analysis.py` | Validate analysis results | `python scripts/validation/validate_latent_space_analysis.py` |

### Setup Validation

Run before starting analysis to ensure all dependencies are installed:

```bash
python scripts/validation/validate_setup.py
```

Checks:
- Python version (3.8+)
- Required packages (numpy, scipy, torch, quspin, etc.)
- QuSpin Hamiltonian construction
- Optional GPU availability

### Laptop Setup Test

Quick test to verify laptop configuration:

```bash
python scripts/validation/test_laptop_setup.py
```

Tests:
- Small ED computation (2×2 lattice)
- Observable calculation
- Memory usage
- Execution time

### Analysis Validation

Validate analysis results after completion:

```bash
python scripts/validation/validate_latent_space_analysis.py
```

Checks:
- Output files exist
- Data format correctness
- Physical consistency
- Numerical quality

## Quick Reference

### First Time Setup

```bash
# 1. Validate installation
python scripts/validation/validate_setup.py

# 2. Test laptop setup (optional)
python scripts/validation/test_laptop_setup.py
```

### Running Analysis

```bash
# Laptop
python scripts/deployment/run_laptop_analysis.py --stage all

# VM (after vm_setup.sh)
python scripts/deployment/run_vm_analysis.py --stage all

# Monitor progress (separate terminal)
python scripts/utilities/monitor_laptop.py
```

### After Analysis

```bash
# Validate results
python scripts/validation/validate_latent_space_analysis.py
```

## Configuration

All scripts use configuration files from `configs/`:
- `configs/default_config.yaml` - Base configuration
- `configs/laptop_config.yaml` - Laptop-optimized (auto-generated)
- `configs/vm_config.yaml` - VM-optimized (auto-generated)

## Logging

All scripts log to:
- Console (INFO level)
- `logs/j1j2_prometheus.log` (DEBUG level)

## Error Handling

Scripts include comprehensive error handling:
- Graceful failures with informative messages
- Automatic checkpointing for resumption
- Resource monitoring and warnings
- Validation at each stage

## Platform Support

| Platform | Supported Scripts |
|----------|-------------------|
| Windows | All Python scripts |
| Linux | All scripts |
| macOS | All scripts |

Note: Shell scripts (`.sh`) require bash on Windows (WSL, Git Bash, or Cygwin).

## Getting Help

For detailed usage:
```bash
python scripts/deployment/run_laptop_analysis.py --help
python scripts/deployment/run_vm_analysis.py --help
```

For troubleshooting, see:
- `LAPTOP_QUICK_START.md` - Laptop deployment guide
- `VM_QUICK_START.md` - VM deployment guide
- `SETUP.md` - Installation guide

## Development

When adding new scripts:
1. Place in appropriate subdirectory
2. Add entry to this README
3. Include `--help` option
4. Add logging configuration
5. Include error handling
6. Update main documentation

## License

See [LICENSE](../LICENSE) file for details.
