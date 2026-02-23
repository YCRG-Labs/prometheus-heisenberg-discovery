#!/usr/bin/env python3
"""
Laptop Progress Monitor

Simple script to monitor analysis progress on laptop.
Shows system resources, stage progress, and recent log entries.
"""

import json
import psutil
import time
from pathlib import Path
from datetime import datetime


def print_header():
    print("\n" + "="*70)
    print("J1-J2 HEISENBERG LAPTOP ANALYSIS MONITOR")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_system_resources():
    print("SYSTEM RESOURCES")
    print("-"*70)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.used / 1024**3:.1f} GB / {memory.total / 1024**3:.1f} GB ({memory.percent:.1f}%)")
    print(f"RAM Available: {memory.available / 1024**3:.1f} GB")
    
    # Disk
    disk = psutil.disk_usage('.')
    print(f"Disk: {disk.used / 1024**3:.1f} GB / {disk.total / 1024**3:.1f} GB ({disk.percent:.1f}%)")
    
    # Temperature (if available)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if 'cpu' in entry.label.lower() or 'core' in entry.label.lower():
                        print(f"CPU Temp: {entry.current}°C")
                        break
    except:
        pass
    
    print()


def print_process_status():
    print("PROCESS STATUS")
    print("-"*70)
    
    # Check if analysis is running
    running = False
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('main_pipeline.py' in str(arg) or 'run_laptop_analysis.py' in str(arg) for arg in cmdline):
                running = True
                print(f"✓ Analysis is RUNNING (PID: {proc.pid})")
                print(f"  CPU: {proc.cpu_percent()}%")
                print(f"  Memory: {proc.memory_info().rss / 1024**3:.1f} GB")
                break
        except:
            pass
    
    if not running:
        print("✗ Analysis is NOT running")
    
    print()


def print_stage_progress():
    print("STAGE PROGRESS")
    print("-"*70)
    
    progress_file = Path("laptop_progress.json")
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            stages = json.load(f)
        
        for stage, info in stages.items():
            status = "✓ COMPLETE" if info['completed'] else "⧗ PENDING"
            print(f"{stage.upper():12} {status}")
            if info['completed']:
                duration = info.get('duration', 0)
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                print(f"             Duration: {hours}h {minutes}m")
                print(f"             Completed: {info['timestamp']}")
    else:
        print("No progress file found. Analysis not started yet.")
    
    print()


def print_checkpoints():
    print("CHECKPOINTS")
    print("-"*70)
    
    # ED checkpoints
    ed_dir = Path("checkpoints/ed_checkpoints")
    if ed_dir.exists():
        ed_files = list(ed_dir.glob("*.pkl"))
        print(f"ED Checkpoints: {len(ed_files)} files")
        if ed_files:
            latest = max(ed_files, key=lambda p: p.stat().st_mtime)
            print(f"  Latest: {latest.name} ({latest.stat().st_size / 1024**2:.1f} MB)")
    else:
        print("ED Checkpoints: None yet")
    
    # Q-VAE models
    qvae_dir = Path("checkpoints/qvae_models")
    if qvae_dir.exists():
        qvae_files = list(qvae_dir.glob("*.pt"))
        print(f"Q-VAE Models: {len(qvae_files)} files")
    else:
        print("Q-VAE Models: None yet")
    
    print()


def print_output_files():
    print("OUTPUT FILES")
    print("-"*70)
    
    output_dir = Path("output")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            print(f"Generated {len(files)} output files:")
            for f in sorted(files)[:10]:  # Show first 10
                size = f.stat().st_size
                if size > 1024**2:
                    size_str = f"{size / 1024**2:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                print(f"  {f.name:40} {size_str:>10}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
        else:
            print("No output files yet")
    else:
        print("Output directory not created yet")
    
    print()


def print_recent_logs():
    print("RECENT LOG ENTRIES (last 10 lines)")
    print("-"*70)
    
    log_file = Path("logs/j1j2_prometheus.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
    else:
        print("No log file found yet")
    
    print()


def print_tips():
    print("TIPS")
    print("-"*70)
    print("• Keep laptop plugged in and well-ventilated")
    print("• Disable sleep mode: powercfg /change standby-timeout-ac 0")
    print("• Check full logs: Get-Content logs/j1j2_prometheus.log -Tail 20 -Wait")
    print("• Stop analysis: Find process in Task Manager and end it")
    print()


def main():
    try:
        print_header()
        print_system_resources()
        print_process_status()
        print_stage_progress()
        print_checkpoints()
        print_output_files()
        print_recent_logs()
        print_tips()
        
        print("="*70)
        print("Monitor will refresh every 60 seconds. Press Ctrl+C to exit.")
        print("="*70)
        
        # Continuous monitoring
        while True:
            time.sleep(60)
            print("\n" * 2)
            print_header()
            print_system_resources()
            print_process_status()
            print_stage_progress()
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
