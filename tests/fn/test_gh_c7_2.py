"""Tests for gh_c7_2.ghosal_kern_mix_kl."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_2 import ghosal_kern_mix_kl


def test_gh_c7_2_basic():
    """Test basic functionality."""
    sd = 0.3
    grids = (3, 9, 27)
    span = 3.0
    result = ghosal_kern_mix_kl(sd=sd, grids=grids, span=span)
    assert "estimate" in result
    assert "kl_by_grid" in result
    assert "improving" in result
    assert "method" in result

    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    assert float(est) >= 0.0

    kl_by_grid = np.asarray(result["kl_by_grid"], dtype=float)
    assert len(kl_by_grid) == len(grids)
    assert np.all(np.isfinite(kl_by_grid))
    assert np.all(kl_by_grid >= 0.0)

    # The estimate returned should equal the last (finest) grid's KL.
    assert np.allclose(float(est), float(kl_by_grid[-1]))

    # The docstring states that finer grids drive KL toward zero; the
    # finest grid should not be worse than the coarsest (up to a
    # tiny numerical tolerance).
    assert bool(result["improving"]) is True
    assert float(kl_by_grid[-1]) <= float(kl_by_grid[0]) + 1e-12


def test_gh_c7_2_edge():
    """Test edge cases: a single-element grid yields a non-negative KL."""
    result = ghosal_kern_mix_kl(sd=0.3, grids=(1,), span=3.0)
    assert "estimate" in result
    assert "kl_by_grid" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(est)
    assert est >= 0.0
