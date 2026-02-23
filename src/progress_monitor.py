"""Progress Monitoring Module

This module provides progress tracking, time estimation, and checkpoint management
for long-running computations in the J1-J2 Heisenberg Prometheus framework.
"""

import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime, timedelta


class ProgressMonitor:
    """Progress monitor for tracking computation progress and estimating remaining time
    
    Features:
    - Progress tracking with percentage completion
    - Time estimation for remaining work
    - Checkpoint management for resumption
    - Logging of progress milestones
    
    Attributes:
        name: Name of the monitored task
        total_items: Total number of items to process
        completed_items: Number of completed items
        start_time: Start time of the task
        logger: Logger instance
        checkpoint_file: Path to checkpoint file
    """
    
    def __init__(
        self,
        name: str,
        total_items: int,
        checkpoint_file: Optional[Path] = None,
        log_interval: int = 10
    ):
        """Initialize ProgressMonitor
        
        Args:
            name: Name of the monitored task
            total_items: Total number of items to process
            checkpoint_file: Optional path to checkpoint file for resumption
            log_interval: Log progress every N items (default: 10)
        """
        self.name = name
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        self.checkpoint_file = checkpoint_file
        self.log_interval = log_interval
        self.last_log_time = self.start_time
        
        # Load checkpoint if available
        if checkpoint_file and checkpoint_file.exists():
            self._load_checkpoint()
        
        self.logger.info(
            f"Progress monitor initialized for '{name}': "
            f"{self.completed_items}/{total_items} items"
        )
    
    def update(self, increment: int = 1) -> None:
        """Update progress by incrementing completed items
        
        Args:
            increment: Number of items completed (default: 1)
        """
        self.completed_items += increment
        
        # Log progress at intervals
        if (self.completed_items % self.log_interval == 0 or 
            self.completed_items == self.total_items):
            self._log_progress()
        
        # Save checkpoint
        if self.checkpoint_file:
            self._save_checkpoint()
    
    def _log_progress(self) -> None:
        """Log current progress with time estimates"""
        elapsed_time = time.time() - self.start_time
        progress_pct = (self.completed_items / self.total_items) * 100
        
        # Estimate remaining time
        if self.completed_items > 0:
            time_per_item = elapsed_time / self.completed_items
            remaining_items = self.total_items - self.completed_items
            estimated_remaining = time_per_item * remaining_items
            
            # Format times
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining)
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            eta_str = eta.strftime("%Y-%m-%d %H:%M:%S")
            
            self.logger.info(
                f"{self.name}: {self.completed_items}/{self.total_items} "
                f"({progress_pct:.1f}%) | "
                f"Elapsed: {elapsed_str} | "
                f"Remaining: ~{remaining_str} | "
                f"ETA: {eta_str}"
            )
        else:
            elapsed_str = self._format_time(elapsed_time)
            self.logger.info(
                f"{self.name}: {self.completed_items}/{self.total_items} "
                f"({progress_pct:.1f}%) | "
                f"Elapsed: {elapsed_str}"
            )
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (e.g., "2h 15m 30s")
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
    
    def _save_checkpoint(self) -> None:
        """Save progress checkpoint to file"""
        if not self.checkpoint_file:
            return
        
        checkpoint_data = {
            'name': self.name,
            'total_items': self.total_items,
            'completed_items': self.completed_items,
            'start_time': self.start_time,
            'last_update': time.time(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure parent directory exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write checkpoint
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _load_checkpoint(self) -> None:
        """Load progress checkpoint from file"""
        if not self.checkpoint_file or not self.checkpoint_file.exists():
            return
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.completed_items = checkpoint_data.get('completed_items', 0)
            self.start_time = checkpoint_data.get('start_time', time.time())
            
            self.logger.info(
                f"Loaded checkpoint for '{self.name}': "
                f"resuming from {self.completed_items}/{self.total_items} items"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information
        
        Returns:
            Dictionary with progress metrics:
                - completed: Number of completed items
                - total: Total number of items
                - percentage: Progress percentage
                - elapsed_time: Elapsed time in seconds
                - estimated_remaining: Estimated remaining time in seconds
        """
        elapsed_time = time.time() - self.start_time
        progress_pct = (self.completed_items / self.total_items) * 100
        
        if self.completed_items > 0:
            time_per_item = elapsed_time / self.completed_items
            remaining_items = self.total_items - self.completed_items
            estimated_remaining = time_per_item * remaining_items
        else:
            estimated_remaining = None
        
        return {
            'completed': self.completed_items,
            'total': self.total_items,
            'percentage': progress_pct,
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining
        }
    
    def is_complete(self) -> bool:
        """Check if all items are completed
        
        Returns:
            True if completed_items >= total_items
        """
        return self.completed_items >= self.total_items
    
    def finalize(self) -> None:
        """Finalize progress monitoring and log summary"""
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed_time)
        
        self.logger.info(
            f"{self.name} completed: {self.completed_items}/{self.total_items} items "
            f"in {elapsed_str}"
        )
        
        # Remove checkpoint file if it exists
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                self.logger.debug(f"Removed checkpoint file: {self.checkpoint_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint file: {e}")


class StepTimer:
    """Simple timer for measuring execution time of pipeline steps
    
    Usage:
        with StepTimer("Step name"):
            # code to time
    """
    
    def __init__(self, step_name: str):
        """Initialize StepTimer
        
        Args:
            step_name: Name of the step being timed
        """
        self.step_name = step_name
        self.logger = logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.step_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration"""
        elapsed_time = time.time() - self.start_time
        
        if exc_type is None:
            # Success
            self.logger.info(
                f"Completed: {self.step_name} "
                f"(duration: {self._format_time(elapsed_time)})"
            )
        else:
            # Error occurred
            self.logger.error(
                f"Failed: {self.step_name} "
                f"(duration: {self._format_time(elapsed_time)})"
            )
        
        return False  # Don't suppress exceptions
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.0f}s"


def log_system_info() -> None:
    """Log system information for reproducibility"""
    import platform
    import sys
    import torch
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CPU count: {torch.get_num_threads()}")
    logger.info("=" * 80)
