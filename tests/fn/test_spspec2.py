"""Tests for spspec2.schabenberger_spectral_sim."""

from morie.fn import _array_core as np

from morie.fn.spspec2 import schabenberger_spectral_sim


def test_spspec2_basic():
    """Spectral root reproduces the covariance and the reported metadata."""
    mu = [0.0, 1.0, 2.0]
    cov = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    result = schabenberger_spectral_sim(mu, cov, seed=7)

    assert result["n"] == 3
    assert result["method"] == "spectral decomposition"

    # The spectral root is symmetric and squares back to Sigma.
    root = np.asarray(result["root"], dtype=float)
    assert root.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            assert abs(root[i][j] - root[j][i]) < 1e-9
    prod = np.asarray(np.dot(root, root), dtype=float)
    for i in range(3):
        for j in range(3):
            assert abs(prod[i][j] - cov[i][j]) < 1e-9

    # Eigenvalues are those of Sigma: they are positive and sum to the trace.
    vals = [float(v) for v in np.asarray(result["eigenvalues"]).ravel()]
    assert len(vals) == 3
    assert min(vals) > 0.0
    assert abs(sum(vals) - 9.0) < 1e-9

    field = [float(v) for v in np.asarray(result["field"]).ravel()]
    assert len(field) == 3
    # Same seed, same stream: the draw is reproducible.
    again = schabenberger_spectral_sim(mu, cov, seed=7)
    for a, b in zip(field, [float(v) for v in np.asarray(again["field"]).ravel()]):
        assert abs(a - b) < 1e-12


def test_spspec2_edge():
    """n = 1: the root is the scalar standard deviation."""
    result = schabenberger_spectral_sim([2.0], [[9.0]], seed=1)
    assert result["n"] == 1
    assert abs(float(np.asarray(result["root"]).ravel()[0]) - 3.0) < 1e-9
    assert abs(float(np.asarray(result["eigenvalues"]).ravel()[0]) - 9.0) < 1e-9

    # A mean/covariance mismatch is rejected.
    try:
        schabenberger_spectral_sim([0.0, 1.0], [[1.0]])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched shapes")
