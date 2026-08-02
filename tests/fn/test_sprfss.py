"""Tests for sprfss: the stationarity hierarchy (Schabenberger & Gotway 2005, Sec 2.2).

Strict > second-order > intrinsic. The screen reports block-wise mean and
variance drift for second-order stationarity, and an increment bias for
intrinsic stationarity -- the latter is about the INCREMENTS,
E[Z(s+h) - Z(s)] = 0, not about the levels.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.sprfss import schabenberger_random_field_stationarity as sprfss


def _lattice(n=40, step=2.0):
    # 40x40 gives 100 points in each of the 16 blocks. Smaller lattices make
    # the block variances themselves noisy enough to trip the variance-drift
    # screen on genuinely stationary noise -- that is sampling variability in
    # the fixture, not a property of the field.
    g = np.arange(n) / step
    return np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)


def _stationary(coords, seed=11):
    rng = np.random.default_rng(seed)
    return rng.normal(size=coords.shape[0])


def test_stationary_field_passes_every_level():
    coords = _lattice()
    res = sprfss(coords, _stationary(coords))
    assert res["mean_stationary"]
    assert res["variance_stationary"]
    assert res["second_order_plausible"]
    assert res["intrinsic_plausible"]


def test_linear_trend_breaks_intrinsic_stationarity():
    """A linear trend leaves the increment VARIANCE flat, so a variance-drift
    screen passes it for the wrong reason. The book's condition is on the
    increment MEAN, which a trend violates outright."""
    coords = _lattice()
    z = _stationary(coords) + 0.8 * coords[:, 0]
    res = sprfss(coords, z)
    assert not res["intrinsic_plausible"]
    assert res["increment_bias"] > sprfss(coords, _stationary(coords))["increment_bias"]


def test_increment_bias_needs_oriented_lags():
    """Binning on lag DISTANCE alone averages the +x and -x pairs together, so
    a trend cancels itself exactly and passes. Orienting each pair into one
    half-space is what makes the trend visible -- assert it actually is."""
    coords = _lattice()
    base = _stationary(coords)
    flat = sprfss(coords, base)["increment_bias"]
    trend = sprfss(coords, base + 0.8 * coords[:, 0])["increment_bias"]
    assert trend > 5.0 * flat


def test_variance_drift_detects_heteroscedasticity():
    """Second-order stationarity requires a constant variance; scaling one
    half of the field must show up as variance drift, not mean drift."""
    coords = _lattice()
    z = _stationary(coords)
    z = np.where(coords[:, 0] > coords[:, 0].mean(), z * 6.0, z)
    res = sprfss(coords, z)
    assert not res["variance_stationary"]
    assert not res["second_order_plausible"]


def test_strict_implies_second_order():
    """The hierarchy is nested: a field flagged strict-if-Gaussian must also
    be second-order plausible. A screen that can report the reverse is wrong."""
    coords = _lattice()
    for seed in (1, 2, 3, 4, 5):
        res = sprfss(coords, _stationary(coords, seed=seed))
        if res["strict_if_gaussian"]:
            assert res["second_order_plausible"]


def test_rejects_bad_input():
    coords = _lattice(n=4)
    with pytest.raises(ValueError):
        sprfss(coords, np.ones(3))
