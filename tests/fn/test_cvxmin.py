"""Tests for cvxmin.boyd_minimum_norm."""

from morie.fn import _array_core as np

from morie.fn.cvxmin import boyd_minimum_norm


def test_cvxmin_basic():
    """Test basic functionality across all supported norms."""
    rng = np.random.default_rng(42)
    # Underdetermined system: more unknowns than equations.
    A = rng.normal(0, 1, (5, 10))
    b = rng.normal(0, 1, 5)

    for norm in (1, 2, "inf"):
        result = boyd_minimum_norm(A, b, norm)

        # The function returns a RichResult with a `payload` dict.
        assert hasattr(result, "payload")
        assert isinstance(result.payload, dict)

        # Documented payload keys.
        for key in ("x", "norm_value", "n_nonzero", "residual",
                    "feasible", "in_row_space", "norm"):
            assert key in result.payload

        # x must have the expected shape and satisfy the constraints.
        x = np.asarray(result.payload["x"], dtype=float)
        assert x.shape == (10,)
        resid = np.asarray(result.payload["residual"], dtype=float)
        assert np.max(np.abs(resid)) < 1e-7
        assert bool(result.payload["feasible"])

        # n_nonzero should match the documented count of non-tiny entries.
        expected_nnz = int(np.sum(np.abs(x) > 1e-8))
        assert int(result.payload["n_nonzero"]) == expected_nnz

        # Reported norm string must echo the requested norm.
        assert result.payload["norm"] == str(norm)


def test_cvxmin_l2_specific():
    """The l2 solution is dense and lies in the row space of A."""
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([3.0])
    result = boyd_minimum_norm(A, b, norm=2)

    x = np.asarray(result.payload["x"], dtype=float)
    # Independent expectation: every coordinate carries 1.0.
    expected = np.array([1.0, 1.0, 1.0])
    assert np.max(np.abs(x - expected)) < 1e-6
    assert bool(result.payload["in_row_space"])


def test_cvxmin_l1_specific():
    """The l1 solution is sparse: one coordinate carries the load."""
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([3.0])
    result = boyd_minimum_norm(A, b, norm=1)

    x = np.asarray(result.payload["x"], dtype=float)
    # Independent expectation: norm_value == sum(|x|) == |b| == 3.0.
    assert abs(float(result.payload["norm_value"]) - 3.0) < 1e-6
    assert int(result.payload["n_nonzero"]) == 1


def test_cvxmin_inf_specific():
    """The l_inf solution spreads the mass as evenly as allowed."""
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([3.0])
    result = boyd_minimum_norm(A, b, norm="inf")

    x = np.asarray(result.payload["x"], dtype=float)
    # Independent expectation: max(|x|) == 1.0 (||x||_inf for x_i == 1).
    assert abs(float(result.payload["norm_value"]) - 1.0) < 1e-6
    assert abs(float(np.max(x)) - float(np.min(x))) < 1e-6
