"""
Unit tests for custom exception classes.
"""

import pytest
from src.exceptions import (
    PrometheusError,
    ComputationError,
    ConvergenceError,
    ValidationError,
    NormalizationError,
    HermitianError
)


class TestPrometheusError:
    """Tests for base PrometheusError class."""
    
    def test_basic_error(self):
        """Test basic error creation without context."""
        error = PrometheusError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.context == {}
    
    def test_error_with_context(self):
        """Test error creation with context information."""
        context = {'param': 'value', 'number': 42}
        error = PrometheusError("Test error", context=context)
        assert error.message == "Test error"
        assert error.context == context
        assert "param=value" in str(error)
        assert "number=42" in str(error)
    
    def test_error_inheritance(self):
        """Test that error can be caught as Exception."""
        with pytest.raises(Exception):
            raise PrometheusError("Test")


class TestComputationError:
    """Tests for ComputationError class."""
    
    def test_computation_error(self):
        """Test computation error creation."""
        error = ComputationError("Matrix operation failed", 
                                context={'operation': 'eigenvalue'})
        assert isinstance(error, PrometheusError)
        assert "Matrix operation failed" in str(error)
        assert "operation=eigenvalue" in str(error)


class TestConvergenceError:
    """Tests for ConvergenceError class."""
    
    def test_convergence_error_basic(self):
        """Test basic convergence error."""
        error = ConvergenceError("Failed to converge")
        assert isinstance(error, PrometheusError)
        assert "Failed to converge" in str(error)
    
    def test_convergence_error_with_iterations(self):
        """Test convergence error with iteration details."""
        error = ConvergenceError("Lanczos failed", iterations=100, residual=1e-5)
        assert error.context['iterations'] == 100
        assert error.context['residual'] == 1e-5
        assert "iterations=100" in str(error)
        assert "residual=" in str(error)
    
    def test_convergence_error_with_context(self):
        """Test convergence error with additional context."""
        error = ConvergenceError("Failed", iterations=50, 
                                context={'j2_j1': 0.5, 'L': 4})
        assert error.context['iterations'] == 50
        assert error.context['j2_j1'] == 0.5
        assert error.context['L'] == 4


class TestValidationError:
    """Tests for ValidationError class."""
    
    def test_validation_error_basic(self):
        """Test basic validation error."""
        error = ValidationError("Invalid parameter")
        assert isinstance(error, PrometheusError)
        assert "Invalid parameter" in str(error)
    
    def test_validation_error_with_expected_actual(self):
        """Test validation error with expected and actual values."""
        error = ValidationError("Value out of range", expected="[0, 1]", actual=1.5)
        assert error.context['expected'] == "[0, 1]"
        assert error.context['actual'] == 1.5
        assert "expected=[0, 1]" in str(error)
        assert "actual=1.5" in str(error)


class TestNormalizationError:
    """Tests for NormalizationError class."""
    
    def test_normalization_error_basic(self):
        """Test basic normalization error."""
        error = NormalizationError("Wavefunction not normalized")
        assert isinstance(error, ValidationError)
        assert "Wavefunction not normalized" in str(error)
    
    def test_normalization_error_with_norm(self):
        """Test normalization error with norm value."""
        error = NormalizationError("Norm violation", norm=1.001, tolerance=1e-8)
        assert error.context['norm'] == 1.001
        assert error.context['tolerance'] == 1e-8
        assert error.context['expected'] == 1.0
        assert error.context['actual'] == 1.001
        assert "norm=1.001" in str(error)
        assert "tolerance=" in str(error)
    
    def test_normalization_error_with_context(self):
        """Test normalization error with additional context."""
        error = NormalizationError("Norm violation", norm=0.99, 
                                  context={'j2_j1': 0.5, 'L': 4})
        assert error.context['norm'] == 0.99
        assert error.context['j2_j1'] == 0.5


class TestHermitianError:
    """Tests for HermitianError class."""
    
    def test_hermitian_error_basic(self):
        """Test basic Hermiticity error."""
        error = HermitianError("Hamiltonian not Hermitian")
        assert isinstance(error, ValidationError)
        assert "Hamiltonian not Hermitian" in str(error)
    
    def test_hermitian_error_with_deviation(self):
        """Test Hermiticity error with deviation value."""
        error = HermitianError("Hermiticity violated", 
                              max_deviation=1e-6, tolerance=1e-10)
        assert error.context['max_deviation'] == 1e-6
        assert error.context['tolerance'] == 1e-10
        assert "max_deviation=" in str(error)
        assert "tolerance=" in str(error)
    
    def test_hermitian_error_with_context(self):
        """Test Hermiticity error with additional context."""
        error = HermitianError("Hermiticity violated", max_deviation=1e-5,
                              context={'j2_j1': 0.5, 'L': 4})
        assert error.context['max_deviation'] == 1e-5
        assert error.context['j2_j1'] == 0.5
        assert error.context['L'] == 4


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catching."""
    
    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        with pytest.raises(NormalizationError):
            raise NormalizationError("Test")
    
    def test_catch_as_validation_error(self):
        """Test catching NormalizationError as ValidationError."""
        with pytest.raises(ValidationError):
            raise NormalizationError("Test")
    
    def test_catch_as_base_error(self):
        """Test catching any custom error as PrometheusError."""
        with pytest.raises(PrometheusError):
            raise ConvergenceError("Test")
    
    def test_catch_as_exception(self):
        """Test catching any custom error as Exception."""
        with pytest.raises(Exception):
            raise ComputationError("Test")
