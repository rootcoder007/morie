"""Tests for hmcsam.hamiltonian_mc."""

import pytest

from morie.fn.hmcsam import hamiltonian_mc


def test_hmcsam_basic():
    """Test basic functionality."""
    def log_p(x):
        return -0.5 * sum(v * v for v in x)
    def grad_log_p(x):
        return [-v for v in x]
    x0 = [1.0, 2.0]
    result = hamiltonian_mc(log_p, grad_log_p, x0, step_size=0.1, L=10, n_iter=10)
    assert isinstance(result, dict)
    assert "estimate" in result


def test_hmcsam_edge():
    """Test edge cases."""
    def log_p(x):
        return -0.5 * sum(v * v for v in x)
    def grad_log_p(x):
        return [-v for v in x]
    x0 = [1.0, 2.0]
    with pytest.raises(ValueError):
        hamiltonian_mc(log_p, grad_log_p, x0, step_size=-0.1, L=10, n_iter=10)
