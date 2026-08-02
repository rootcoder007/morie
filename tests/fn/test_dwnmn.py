"""dwnmn: dynamic W-NOMINATE / time-varying ideal points.

Armstrong et al., Ch 6 (Bayesian Scaling Models, printed p.181). Ideal points
are smoothed across periods with an evolution variance sigma_w.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.dwnmn import dynamic_wnominate as dw


def test_dwnmn_smoothing_reduces_period_to_period_variation():
    """The whole point: raw per-period estimates jump around, the smoothed
    path should not."""
    rng = np.random.default_rng(3501)
    truth = np.cumsum(rng.normal(0, 0.1, 20))
    raw = truth + rng.normal(0, 0.8, (5, 20))
    r = dw(raw, sigma_w=0.1)
    sm = np.asarray(r["smoothed"])
    assert np.mean(np.std(np.diff(sm, axis=1), axis=1)) < np.mean(
        np.std(np.diff(raw, axis=1), axis=1)
    )


def test_dwnmn_a_larger_sigma_w_smooths_less():
    """sigma_w is how much movement the model expects between periods, so a
    bigger value must track the raw series more closely."""
    rng = np.random.default_rng(3511)
    raw = rng.standard_normal((4, 25))
    tight = np.asarray(dw(raw, sigma_w=0.01)["smoothed"])
    loose = np.asarray(dw(raw, sigma_w=100.0)["smoothed"])
    assert np.mean((loose - raw) ** 2) < np.mean((tight - raw) ** 2)


def test_dwnmn_error_is_U_shaped_in_sigma_w():
    """Bias-variance in one picture, and a much stronger check than "beats
    the raw series at one arbitrary sigma_w".

    Too small a sigma_w assumes the ideal point barely moves and over-smooths
    a genuinely moving one; too large lets the noise straight through. There
    must therefore be an interior minimum. Measured on a sine path with
    sigma = 0.5 noise over 30 periods:

        raw            0.2851
        sigma_w 0.05   0.3540   <- over-smoothed, WORSE than raw
        sigma_w 0.20   0.1022
        sigma_w 0.50   0.0527   <- best
        sigma_w 1.00   0.0806
        sigma_w 5.00   0.2519   <- under-smoothed, approaching raw
    """
    rng = np.random.default_rng(3517)
    t = np.linspace(0, 1, 30)
    truth = np.vstack([np.sin(2 * np.pi * t), -np.sin(2 * np.pi * t)])
    raw = truth + rng.normal(0, 0.5, truth.shape)
    grid = [0.05, 0.2, 0.5, 1.0, 5.0]
    mse = [float(np.mean((np.asarray(dw(raw, sigma_w=s)["smoothed"]) - truth) ** 2))
           for s in grid]
    best = int(np.argmin(mse))
    assert 0 < best < len(grid) - 1, f"expected an interior optimum, got {mse}"
    assert mse[best] < float(np.mean((raw - truth) ** 2)) / 3


def test_dwnmn_shapes_and_counts_are_reported():
    rng = np.random.default_rng(3521)
    r = dw(rng.standard_normal((6, 12)), sigma_w=0.2)
    assert r["n_units"] == 6 and r["n_periods"] == 12
    assert np.asarray(r["smoothed"]).shape == (6, 12)
    assert np.asarray(r["raw"]).shape == (6, 12)


def test_dwnmn_a_constant_series_is_left_alone():
    """Nothing to smooth: a flat path must come back flat."""
    raw = np.tile(np.array([[1.0], [-1.0]]), (1, 15))
    sm = np.asarray(dw(raw, sigma_w=0.1)["smoothed"])
    assert sm == pytest.approx(raw, abs=1e-6)


def test_dwnmn_echoes_sigma_w():
    rng = np.random.default_rng(3527)
    assert dw(rng.standard_normal((3, 8)), sigma_w=0.33)["sigma_w"] == pytest.approx(0.33)
