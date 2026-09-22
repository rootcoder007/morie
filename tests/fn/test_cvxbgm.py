"""Tests for cvxbgm.boyd_basis_pursuit."""

from morie.fn import _array_core as np

from morie.fn.cvxbgm import boyd_basis_pursuit


def test_cvxbgm_basic():
    """Test basic functionality with the documented example."""
    # From the function's docstring: A is (m, n), b is length m, eps is a scalar.
    A = np.array([[1.0, 0.0, 0.8], [0.0, 1.0, 0.8]])
    b = np.array([1.0, 1.0])
    eps = 0.0
    result = boyd_basis_pursuit(A, b, eps)

    # The function returns a RichResult (dict-like) with the documented keys.
    assert "x" in result
    assert "l1_norm" in result
    assert "residual" in result
    assert "residual_norm" in result
    assert "n_nonzero" in result
    assert "support" in result
    assert "feasible" in result
    assert "trivial" in result

    # Independently computed expectation for this input.
    # The third column is (0.8, 0.8); to hit (1, 1) exactly we need x_3 = 1.25.
    expected_x = np.array([0.0, 0.0, 1.25])
    x = np.asarray(result["x"], dtype=float)
    assert x.shape == (3,)
    for xi, ei in zip(x.ravel(), expected_x.ravel()):
        assert abs(float(xi) - float(ei)) < 1e-6

    # l1_norm = 1.25 by formula: |0| + |0| + |1.25|
    expected_l1 = 0.0 + 0.0 + 1.25
    assert abs(float(result["l1_norm"]) - expected_l1) < 1e-6

    # Support is the index set of nonzero entries of x.
    support = [int(i) for i in result["support"]]
    assert support == [2]

    # n_nonzero is the count of nonzero entries.
    assert int(result["n_nonzero"]) == 1

    # With eps = 0 we demand exact reconstruction, so residual_norm = 0
    # by formula: ||A @ x - b||_2 with x = (0, 0, 1.25) yields (0, 0).
    assert float(result["residual_norm"]) < 1e-8


def test_cvxbgm_edge():
    """Test edge case: eps large enough that the zero solution is optimal."""
    # From the docstring: once eps >= ||b||_2, x = 0 is feasible and trivial.
    A = np.array([[1.0, 0.0, 0.8], [0.0, 1.0, 0.8]])
    b = np.array([1.0, 1.0])
    # ||b||_2 = sqrt(2) ~ 1.414, so eps = 2.0 is past the budget threshold.
    eps = 2.0
    result = boyd_basis_pursuit(A, b, eps)

    assert "x" in result
    assert "n_nonzero" in result
    assert "trivial" in result

    # Independent expectation: x = 0 vector, n_nonzero = 0, trivial flag True.
    x = np.asarray(result["x"], dtype=float)
    assert x.shape == (3,)
    for xi in x.ravel():
        assert abs(float(xi)) < 1e-8

    assert int(result["n_nonzero"]) == 0
    assert bool(result["trivial"]) is True
