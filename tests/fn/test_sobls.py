"""sobls: Sobol low-discrepancy sequence."""

import numpy as np
import pytest

from morie.fn.sobls import sobol_sequence as sob


def test_sobls_shape_and_support():
    r = sob(N=64, d=3, seed=1)
    s = np.asarray(r["sample"])
    assert s.shape == (64, 3)
    assert np.all((s >= 0) & (s <= 1))
    assert r["N"] == 64 and r["d"] == 3


def test_sobls_beats_iid_on_integration_error():
    """The point of a low-discrepancy sequence: quasi-Monte Carlo error falls
    faster than the 1/sqrt(N) of plain random sampling."""
    rng = np.random.default_rng(7)
    N, d = 1024, 3
    f = lambda p: np.exp(-(p**2).sum(axis=1))
    truth_mc = np.mean(f(rng.random((400_000, d))))
    q = np.asarray(sob(N=N, d=d, scramble=True, seed=1)["sample"])
    qmc_err = abs(np.mean(f(q)) - truth_mc)
    mc_err = np.mean([abs(np.mean(f(rng.random((N, d)))) - truth_mc) for _ in range(30)])
    assert qmc_err < mc_err


def test_sobls_is_more_uniform_than_iid_by_star_discrepancy_proxy():
    """Count points in the lower-left box [0,q]^d and compare with q^d."""
    N, d, q = 256, 2, 0.5
    s = np.asarray(sob(N=N, d=d, scramble=True, seed=3)["sample"])
    frac = float(np.mean(np.all(s <= q, axis=1)))
    assert abs(frac - q**d) < 0.02


def test_sobls_margins_are_close_to_uniform_mean():
    s = np.asarray(sob(N=512, d=4, scramble=True, seed=5)["sample"])
    assert np.allclose(s.mean(axis=0), 0.5, atol=0.01)


def test_sobls_unscrambled_is_deterministic_and_seed_free():
    """The plain Sobol sequence is a fixed construction: no seed changes it."""
    a = np.asarray(sob(N=32, d=2, scramble=False, seed=1)["sample"])
    b = np.asarray(sob(N=32, d=2, scramble=False, seed=999)["sample"])
    assert a == pytest.approx(b, abs=0.0)


def test_sobls_scrambling_is_reproducible_but_seed_sensitive():
    a = np.asarray(sob(N=32, d=2, scramble=True, seed=11)["sample"])
    b = np.asarray(sob(N=32, d=2, scramble=True, seed=11)["sample"])
    c = np.asarray(sob(N=32, d=2, scramble=True, seed=12)["sample"])
    assert a == pytest.approx(b, abs=0.0)
    assert not np.allclose(a, c)
