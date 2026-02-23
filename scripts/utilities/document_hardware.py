#!/usr/bin/env python
"""
Hardware Specification Documentation Script

This script documents the hardware and software environment used for running
the J1-J2 Heisenberg Prometheus framework analysis. Run this script to generate
a hardware_specs.json file that can be included with results for reproducibility.

Usage:
    python document_hardware.py
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def get_cpu_info():
    """Get CPU information."""
    try:
        import psutil
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'processor': platform.processor(),
        }
    except ImportError:
        cpu_info = {
            'physical_cores': None,
            'logical_cores': None,
            'max_frequency_mhz': None,
            'processor': platform.processor(),
        }
    return cpu_info


def get_memory_info():
    """Get memory information."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        memory_info = {
            'total_gb': round(mem.total / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2),
        }
    except ImportError:
        memory_info = {
            'total_gb': None,
            'available_gb': None,
        }
    return memory_info


def get_gpu_info():
    """Get GPU information."""
    gpu_info = {'available': False}
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['devices'] = []
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'name': torch.cuda.get_device_name(i),
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'total_memory_gb': round(device_props.total_memory / (1024**3), 2),
                    'multi_processor_count': device_props.multi_processor_count,
                })
            
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['cudnn_version'] = torch.backends.cudnn.version()
    except ImportError:
        pass
    
    return gpu_info


def get_python_info():
    """Get Python environment information."""
    python_info = {
        'version': sys.version,
        'version_info': {
            'major': sys.version_info.major,
            'minor': sys.version_info.minor,
            'micro': sys.version_info.micro,
        },
        'implementation': platform.python_implementation(),
        'compiler': platform.python_compiler(),
    }
    return python_info


def get_package_versions():
    """Get versions of key packages."""
    packages = {}
    
    package_list = [
        'numpy', 'scipy', 'torch', 'pandas', 'scikit-learn',
        'matplotlib', 'seaborn', 'h5py', 'pydantic', 'quspin',
        'pytest', 'hypothesis'
    ]
    
    for package_name in package_list:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            packages[package_name] = version
        except ImportError:
            packages[package_name] = 'not installed'
    
    return packages


def get_os_info():
    """Get operating system information."""
    os_info = {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'platform': platform.platform(),
    }
    return os_info


def document_hardware():
    """Generate complete hardware and software documentation."""
    
    print("Documenting hardware and software environment...")
    print("=" * 60)
    
    specs = {
        'timestamp': datetime.now().isoformat(),
        'os': get_os_info(),
        'python': get_python_info(),
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'gpu': get_gpu_info(),
        'packages': get_package_versions(),
    }
    
    # Print summary
    print("\nOperating System:")
    print(f"  {specs['os']['system']} {specs['os']['release']}")
    
    print("\nPython:")
    print(f"  Version: {specs['python']['version_info']['major']}."
          f"{specs['python']['version_info']['minor']}."
          f"{specs['python']['version_info']['micro']}")
    
    print("\nCPU:")
    if specs['cpu']['physical_cores']:
        print(f"  Physical cores: {specs['cpu']['physical_cores']}")
        print(f"  Logical cores: {specs['cpu']['logical_cores']}")
    print(f"  Processor: {specs['cpu']['processor']}")
    
    print("\nMemory:")
    if specs['memory']['total_gb']:
        print(f"  Total: {specs['memory']['total_gb']} GB")
        print(f"  Available: {specs['memory']['available_gb']} GB")
    
    print("\nGPU:")
    if specs['gpu']['available']:
        print(f"  Available: Yes")
        print(f"  Device count: {specs['gpu']['device_count']}")
        for i, device in enumerate(specs['gpu']['devices']):
            print(f"  Device {i}: {device['name']}")
            print(f"    Memory: {device['total_memory_gb']} GB")
            print(f"    Compute capability: {device['compute_capability']}")
        print(f"  CUDA version: {specs['gpu']['cuda_version']}")
    else:
        print(f"  Available: No")
    
    print("\nKey Package Versions:")
    for package, version in sorted(specs['packages'].items()):
        print(f"  {package:20s}: {version}")
    
    # Save to file
    output_file = Path('hardware_specs.json')
    with open(output_file, 'w') as f:
        json.dump(specs, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Hardware specifications saved to: {output_file}")
    print("Include this file with your results for reproducibility.")
    
    return specs


if __name__ == '__main__':
    try:
        specs = document_hardware()
        sys.exit(0)
    except Exception as e:
        print(f"\nError documenting hardware: {e}", file=sys.stderr)
        sys.exit(1)
