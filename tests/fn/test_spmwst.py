"""spmwst -- Haas moving windows, Schabenberger & Gotway Sec. 8.3.1."""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_nonstat import haas_window
from morie.fn.spmwst import schabenberger_moving_window as mw


def _field(n=200, seed=7):
    rs = np.random.RandomState(seed)
    s = rs.uniform(0, 10, (n, 2))
    z = np.sin(s[:, 0] / 2.0) + 0.3 * rs.standard_normal(n)
    return s, z


def test_haas_window_rule_35_then_5():
    s, _ = _field()
    w = haas_window(s, np.array([5.0, 5.0]))
    assert w["n_sites"] >= 35
    assert (w["n_sites"] - 35) % 5 == 0
    assert w["all_lag_classes_filled"]


def test_window_radius_covers_exactly_its_sites():
    s, _ = _field()
    w = haas_window(s, np.array([5.0, 5.0]))
    d = np.linalg.norm(s - np.array([5.0, 5.0]), axis=1)
    assert int((d <= w["radius"] + 1e-12).sum()) == w["n_sites"]


def test_moving_window_reestimates_theta_locally():
    s, z = _field()
    tg = np.array([[3.0, 3.0], [7.0, 7.0], [5.0, 2.0]])
    r = mw(s, z, targets=tg, min_sites=35)
    assert r["local_variograms"].shape == (3, 2)
    assert len(set(np.round(r["local_variograms"][:, 1], 6))) > 1
    assert r["theta_is_global"] is False


def test_local_kriging_keeps_one_global_theta():
    """p. 425: all n points estimate theta; only the solve is local."""
    s, z = _field()
    tg = np.array([[3.0, 3.0], [7.0, 7.0], [5.0, 2.0]])
    r = mw(s, z, targets=tg, min_sites=35, local_variogram=False)
    assert r["theta_is_global"] is True
    assert len(set(np.round(r["local_variograms"][:, 1], 9))) == 1
    assert r["local_variograms"][0, 1] == pytest.approx(r["global_range"])


def test_prediction_tracks_a_smooth_signal():
    s, z = _field(300, 9)
    tg = s[:20]
    r = mw(s, z, targets=tg, min_sites=35)
    truth = np.sin(tg[:, 0] / 2.0)
    assert float(np.mean(np.abs(r["prediction"] - truth))) < 0.35


def test_default_targets_are_the_observed_sites():
    s, z = _field(60, 3)
    r = mw(s, z, min_sites=20)
    assert r["prediction"].shape == (60,)


def test_caveats_from_the_book_are_reported():
    s, z = _field(80, 5)
    r = mw(s, z, targets=s[:2], min_sites=20)
    assert "no longer best" in r["caveats"]
    assert "spurious discontinuities" in r["caveats"]


def test_tiny_fixed_window_warns_against_the_35_site_rule():
    s, z = _field()
    r = mw(s, z, targets=s[:2], window_size=0.3)
    assert "warning" in r
    assert "35" in r["warning"]


def test_generous_fixed_window_reports_counts():
    s, z = _field()
    r = mw(s, z, targets=s[:2], window_size=6.0)
    assert "fixed_window_counts" in r
    assert np.all(r["fixed_window_counts"] >= 35)


def test_convergence_is_reported_per_window():
    s, z = _field()
    r = mw(s, z, targets=s[:4], min_sites=35)
    assert r["converged"].shape == (4,)
    assert r["converged"].dtype == bool


def test_rejects_mismatched_lengths_and_bad_window():
    s, z = _field(50, 2)
    with pytest.raises(ValueError):
        mw(s, z[:-1])
    with pytest.raises(ValueError):
        mw(s, z, window_size=0.0)
    with pytest.raises(ValueError):
        haas_window(s, s[0], min_sites=1)
