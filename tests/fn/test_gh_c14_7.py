"""Tests for gh_c14_7.ghosal_ssp_mix."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_7 import ghosal_ssp_mix


def test_gh_c14_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.2, 0.3, 0.5])
    atoms = np.array([0.0, 2.5, 5.0])
    result = ghosal_ssp_mix(x, weights, atoms, kernel_sd=0.3)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Independently compute the expected density using the documented
    # formula f(x) = sum_i w_i * N(x; t_i, sd) for each x.
    import math
    sd = 0.3
    w = np.asarray(weights, dtype=float)
    t = np.asarray(atoms, dtype=float)
    norm = w.sum()
    w = w / norm
    def npdf(v, m):
        z = (v - m) / sd
        return math.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi))
    expected = [
        sum(float(pi) * npdf(float(v), float(tt)) for pi, tt in zip(w, t))
        for v in np.asarray(x, dtype=float)
    ]
    got = np.asarray(result["density"], dtype=float)
    assert np.allclose(got, expected, atol=1e-12)


def test_gh_c14_7_edge():
    """Test edge cases."""
    weights = np.array([1.0])
    atoms = np.array([42.0])
    result = ghosal_ssp_mix(np.array([42.0]), weights, atoms)
    # Result is a density on a single point, exposed via the documented
    # "density" / "estimate" keys.
    assert np.isfinite(float(result["estimate"]))
