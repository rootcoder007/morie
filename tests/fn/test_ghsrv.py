"""Tests for ghsrv.ghosal_survival_beta_process."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ghsrv import ghosal_survival_beta_process


def test_ghsrv_survival_curve_is_monotone_in_unit_interval():
    rng = np.random.default_rng(0)
    r = ghosal_survival_beta_process(rng.exponential(2.0, 200))
    s = np.asarray(r["S_post"], dtype=float)
    assert np.all(s >= -1e-12) and np.all(s <= 1 + 1e-12)
    assert np.all(np.diff(s) <= 1e-9)


def test_ghsrv_tracks_the_true_exponential_survival():
    rng = np.random.default_rng(1)
    r = ghosal_survival_beta_process(rng.exponential(1.0, 500))
    times = np.asarray(r["times"], dtype=float)
    s = np.asarray(r["S_post"], dtype=float)
    i = np.searchsorted(times, 1.0)
    assert 0 < i < s.size
    assert s[i] == pytest.approx(np.exp(-1.0), abs=0.08)


def test_ghsrv_censoring_raises_the_tail():
    t = np.linspace(0.5, 4.0, 40)
    full = ghosal_survival_beta_process(t, event=np.ones(40, dtype=int))
    cens = ghosal_survival_beta_process(t, event=(np.arange(40) < 10).astype(int))
    s_full = np.asarray(full["S_post"], dtype=float)
    s_cens = np.asarray(cens["S_post"], dtype=float)
    assert float(s_cens[-1]) > float(s_full[-1])
