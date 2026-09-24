"""Tests for bmtme.bmtme_model."""

from morie.fn import _array_core as np

from morie.fn.bmtme import bmtme_model


def _to_float_matrix(rng, n_rows, n_cols):
    """Build an n_rows-by-n_cols matrix of Python floats."""
    M = rng.normal(0, 1, (n_rows, n_cols))
    return [[float(x) for x in row] for row in M]


def _to_float_symmetric(rng, J):
    """Build a J-by-J symmetric positive-definite matrix of Python floats."""
    M = rng.normal(0, 1, (J, J))
    G = []
    for i in range(J):
        row = []
        for j in range(J):
            row.append(float((M[i][j] + M[j][i]) / 2))
        G.append(row)
    for i in range(J):
        G[i][i] += float(J)
    return G


def test_bmtme_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    I = 2
    J = 5
    n_T = 2
    N = I * J

    Y = _to_float_matrix(rng, N, n_T)
    G = _to_float_symmetric(rng, J)

    n_iter = 10
    result = bmtme_model(Y, G, I, n_iter=n_iter)

    assert isinstance(result, dict)
    expected_keys = (
        "estimate", "gebv", "b1", "b2", "sigma_g",
        "Sigma_T", "Sigma_E", "R", "mu",
    )
    for key in expected_keys:
        assert key in result

    gebv = result["gebv"]
    assert len(gebv) == N
    for row in gebv:
        assert len(row) == n_T


def test_bmtme_edge():
    """Test with a small but valid configuration."""
    rng = np.random.default_rng(43)
    I = 2
    J = 3
    n_T = 1
    N = I * J

    Y = _to_float_matrix(rng, N, n_T)
    G = _to_float_symmetric(rng, J)

    n_iter = 5
    result = bmtme_model(Y, G, I, n_iter=n_iter)

    assert isinstance(result, dict)
    expected_keys = (
        "estimate", "gebv", "b1", "b2", "sigma_g",
        "Sigma_T", "Sigma_E", "R", "mu",
    )
    for key in expected_keys:
        assert key in result

    gebv = result["gebv"]
    assert len(gebv) == N
    for row in gebv:
        assert len(row) == n_T
