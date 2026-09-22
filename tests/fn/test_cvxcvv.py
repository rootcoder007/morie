"""Tests for cvxcvv.boyd_cvxlin_complement."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.cvxcvv import boyd_cvxlin_complement


def test_cvxcvv_basic():
    """Test basic functionality with a positive definite block."""
    # A: 2x2 SPD, B: 2x1, C: 1x1 scalar -- from the docstring example.
    A = [[2.0, 0.0], [0.0, 2.0]]
    B = [[1.0], [0.0]]
    C = [[1.0]]

    result = boyd_cvxlin_complement(A, B, C)

    # The function returns a RichResult that supports dict-style access.
    assert isinstance(result, dict) or hasattr(result, "__getitem__")

    # All three Schur-complement conditions hold: A psd, Schur psd,
    # and the range condition (vacuous here since A is invertible).
    assert bool(result["psd"]) is True
    assert bool(result["A_psd"]) is True
    assert bool(result["schur_psd"]) is True
    assert bool(result["range_condition"]) is True

    # Schur complement: C - B^T A^{-1} B = 1 - 0.5 = 0.5, computed
    # independently from the documented formula.
    A_arr = np.asarray(A, dtype=float)
    B_arr = np.asarray(B, dtype=float)
    C_arr = np.asarray(C, dtype=float)
    expected_schur = C_arr - B_arr.T @ np.linalg.inv(A_arr) @ B_arr
    got_schur = np.asarray(result["schur_complement"], dtype=float)
    assert got_schur.shape == expected_schur.shape
    assert np.allclose(got_schur, expected_schur)

    # The full block matrix must be PSD (its smallest eigenvalue >= 0
    # up to tolerance), which implies all three conditions.
    block = np.block([[A_arr, B_arr], [B_arr.T, C_arr]])
    block = 0.5 * (block + block.T)
    expected_min_eig = float(np.linalg.eigvalsh(block).min())
    assert np.isclose(float(result["min_eigenvalue"]), expected_min_eig)


def test_cvxcvv_edge():
    """Test edge case: singular A where range condition is decisive."""
    # A is singular (rank 1), B has a component outside A's range,
    # so the range condition fails and the whole test fails even
    # though the per-block PSD checks might individually pass.
    A = [[1.0, 0.0], [0.0, 0.0]]
    B = [[0.0], [1.0]]
    C = [[1.0]]

    result = boyd_cvxlin_complement(A, B, C)

    assert isinstance(result, dict) or hasattr(result, "__getitem__")

    # A itself is PSD (eigenvalues 1 and 0).
    assert bool(result["A_psd"]) is True
    # But the range condition (I - A A^+) B = 0 is violated.
    assert bool(result["range_condition"]) is False
    # Therefore the overall PSD verdict is False.
    assert bool(result["psd"]) is False
