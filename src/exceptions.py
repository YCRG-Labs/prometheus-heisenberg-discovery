"""
Custom exception classes for the J1-J2 Heisenberg Prometheus framework.

This module defines specialized exceptions for different types of errors
that can occur during computation, providing context information for debugging.
"""

from typing import Optional, Dict, Any


class PrometheusError(Exception):
    """Base exception class for all Prometheus framework errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional context.
        
        Args:
            message: Human-readable error description
            context: Dictionary containing contextual information about the error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return formatted error message with context."""
        base_msg = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_msg} [Context: {context_str}]"
        return base_msg


class ComputationError(PrometheusError):
    """
    Exception raised when a computation fails or produces invalid results.
    
    This includes failures in numerical algorithms, matrix operations,
    or other computational procedures.
    """
    pass


class ConvergenceError(PrometheusError):
    """
    Exception raised when an iterative algorithm fails to converge.
    
    This includes Lanczos algorithm failures, optimization failures,
    or any iterative procedure that doesn't reach convergence criteria.
    """
    
    def __init__(self, message: str, iterations: Optional[int] = None,
                 residual: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize convergence error with iteration details.
        
        Args:
            message: Human-readable error description
            iterations: Number of iterations completed before failure
            residual: Final residual or error measure
            context: Additional contextual information
        """
        context = context or {}
        if iterations is not None:
            context['iterations'] = iterations
        if residual is not None:
            context['residual'] = residual
        super().__init__(message, context)


class ValidationError(PrometheusError):
    """
    Exception raised when data validation fails.
    
    This includes parameter range violations, data consistency checks,
    or any validation that detects invalid input or state.
    """
    
    def __init__(self, message: str, expected: Optional[Any] = None,
                 actual: Optional[Any] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize validation error with expected/actual values.
        
        Args:
            message: Human-readable error description
            expected: Expected value or condition
            actual: Actual value that failed validation
            context: Additional contextual information
        """
        context = context or {}
        if expected is not None:
            context['expected'] = expected
        if actual is not None:
            context['actual'] = actual
        super().__init__(message, context)


class NormalizationError(ValidationError):
    """
    Exception raised when wavefunction normalization is violated.
    
    Quantum wavefunctions must satisfy <ψ|ψ> = 1. This exception is raised
    when this constraint is violated beyond acceptable tolerance.
    """
    
    def __init__(self, message: str, norm: Optional[float] = None,
                 tolerance: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize normalization error with norm details.
        
        Args:
            message: Human-readable error description
            norm: Actual norm value <ψ|ψ>
            tolerance: Tolerance threshold that was violated
            context: Additional contextual information
        """
        context = context or {}
        if norm is not None:
            context['norm'] = norm
        if tolerance is not None:
            context['tolerance'] = tolerance
        super().__init__(message, expected=1.0, actual=norm, context=context)


class HermitianError(ValidationError):
    """
    Exception raised when Hamiltonian Hermiticity is violated.
    
    Physical Hamiltonians must be Hermitian (H = H†). This exception is raised
    when this property is violated beyond acceptable tolerance.
    """
    
    def __init__(self, message: str, max_deviation: Optional[float] = None,
                 tolerance: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize Hermiticity error with deviation details.
        
        Args:
            message: Human-readable error description
            max_deviation: Maximum deviation from Hermiticity
            tolerance: Tolerance threshold that was violated
            context: Additional contextual information
        """
        context = context or {}
        if max_deviation is not None:
            context['max_deviation'] = max_deviation
        if tolerance is not None:
            context['tolerance'] = tolerance
        super().__init__(message, expected='Hermitian', actual=f'deviation={max_deviation}', 
                        context=context)
