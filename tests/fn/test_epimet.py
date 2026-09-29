"""Tests for epimet.epinow2."""

from morie.fn import _array_core as np
from morie.fn.epimet import epinow2


def test_epimet_basic():
    """Test basic functionality."""
    incidence = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    gen_int = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = epinow2(incidence, gen_int)
    assert isinstance(result, dict)
    assert "rt" in result


def test_epimet_edge():
    """Test edge cases."""
    incidence = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    gen_int = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = epinow2(incidence, gen_int)
    assert isinstance(result, dict)


def test_renewal_rt_recomputed():
    """R_t = I_t / sum_k g(k) I_{t-k}, after the mean-delay back-shift."""
    import pytest

    inc = [5.0, 8.0, 12.0, 15.0, 20.0, 24.0, 22.0, 30.0]
    g = [0.2, 0.5, 0.3]
    d = [0.0, 1.0]  # mean delay 1 -> shift by one step
    inf = inc[1:]
    rt = [inf[t] / sum(g[k] * inf[t - k - 1] for k in range(3)) for t in range(3, len(inf))]
    r = epinow2(inc, g, delays=d)
    assert r["shift"] == 1
    assert r["rt"] == pytest.approx(rt, rel=1e-14)
    assert r["mean_rt"] == pytest.approx(sum(rt) / len(rt), rel=1e-14)
