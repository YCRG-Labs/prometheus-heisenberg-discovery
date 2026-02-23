"""Unit tests for logging infrastructure"""

import pytest
import logging
import tempfile
from pathlib import Path

from src.logging_config import setup_logging, get_logger, LoggerMixin


class TestLoggingSetup:
    """Test logging setup functionality"""
    
    def test_setup_logging_console_only(self):
        """Test setting up console-only logging"""
        logger = setup_logging(level="INFO")
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1
    
    def test_setup_logging_with_file(self):
        """Test setting up logging with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(level="DEBUG", log_file=str(log_file))
            
            assert logger.level == logging.DEBUG
            assert log_file.exists()
            
            # Test logging works
            logger.info("Test message")
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_setup_logging_creates_directory(self):
        """Test that logging setup creates log directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs" / "subdir" / "test.log"
            setup_logging(log_file=str(log_file))
            
            assert log_file.parent.exists()
            assert log_file.exists()
    
    def test_get_logger(self):
        """Test getting a named logger"""
        logger = get_logger("test_module")
        assert logger.name == "test_module"
        assert isinstance(logger, logging.Logger)


class TestLoggerMixin:
    """Test LoggerMixin functionality"""
    
    def test_logger_mixin(self):
        """Test LoggerMixin provides logger property"""
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        assert hasattr(obj, 'logger')
        assert isinstance(obj.logger, logging.Logger)
        assert obj.logger.name == "TestClass"
    
    def test_logger_mixin_caching(self):
        """Test that logger is cached"""
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        logger1 = obj.logger
        logger2 = obj.logger
        assert logger1 is logger2
