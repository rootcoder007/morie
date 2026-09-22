"""Tests for DW-NOMINATE."""

from morie.fn import _array_core as np

from morie.fn.dwnom import dwnom


def test_dwnom_smoke():
    rng = np.random.default_rng(42)
    # DW-NOMINATE requires legislators (rows) < voters (cols) and enough
    # complete rows to estimate parameters. Build a small but well-shaped
    # matrix: 13 legislators x 20 votes, fully observed (no NaNs), binary.
    n_legs, n_votes = 13, 20
    votes = (rng.random((n_legs, n_votes)) > 0.4).astype(float)
    r = dwnom(votes, n_dims=1, max_iter=10)
    assert r.name == "dw_nominate_estimate"
    assert "ideal_points" in r.extra
    assert r.extra["gmp"] > 0.0


def test_cheatsheet():
    from morie.fn.dwnom import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
