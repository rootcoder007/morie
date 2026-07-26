"""ssmod: local-level state-space model via Kalman filter + smoother.

Kalman, R. E. (1960), *J. Basic Eng.* 82(1):35-45 -- in the library.

    mu_t = mu_{t-1} + eta_t,  y_t = mu_t + eps_t
"""

import numpy as np
import pytest

from morie.fn.ssmod import state_space_model as ss


def test_ssmod_tracks_a_random_walk_signal():
    rng = np.random.default_rng(3101)
    mu = np.cumsum(rng.normal(0, 0.3, 400))
    y = mu + rng.normal(0, 1.0, 400)
    f = np.asarray(ss(y)["filtered_state"])
    assert np.corrcoef(f, mu)[0, 1] > 0.9


def test_ssmod_filtered_state_is_smoother_than_the_data():
    """The point of filtering: the state path must vary less than the noisy
    observations it was extracted from."""
    rng = np.random.default_rng(3109)
    mu = np.cumsum(rng.normal(0, 0.2, 300))
    y = mu + rng.normal(0, 1.5, 300)
    f = np.asarray(ss(y)["filtered_state"])
    assert np.std(np.diff(f)) < np.std(np.diff(y))


def test_ssmod_smoothed_beats_filtered_on_a_known_signal():
    """The smoother uses the whole sample, the filter only the past, so the
    smoothed path must sit closer to the truth."""
    rng = np.random.default_rng(3119)
    mu = np.cumsum(rng.normal(0, 0.25, 500))
    y = mu + rng.normal(0, 1.0, 500)
    r = ss(y)
    ef = np.mean((np.asarray(r["filtered_state"]) - mu) ** 2)
    es = np.mean((np.asarray(r["smoothed_state"]) - mu) ** 2)
    assert es < ef


def test_ssmod_estimates_a_larger_Q_when_the_state_moves_more():
    """Q is the state-innovation variance; a fast-moving level must give a
    larger Q than a nearly-constant one."""
    rng = np.random.default_rng(3121)
    n = 400
    still = np.zeros(n) + rng.normal(0, 1.0, n)
    movy = np.cumsum(rng.normal(0, 1.0, n)) + rng.normal(0, 1.0, n)
    assert ss(movy)["Q"] > ss(still)["Q"]


def test_ssmod_variances_are_non_negative_and_loglik_is_finite():
    rng = np.random.default_rng(3137)
    r = ss(rng.standard_normal(200))
    assert r["Q"] >= 0 and r["R"] >= 0
    assert np.isfinite(r["loglik"])
    assert r["n"] == 200
    assert np.all(np.asarray(r["filtered_state_variance"]) >= 0)


def test_ssmod_a_constant_series_is_recovered_exactly():
    """No noise, no movement: the level is the constant itself."""
    y = np.full(100, 7.0)
    f = np.asarray(ss(y)["filtered_state"])
    assert f[-1] == pytest.approx(7.0, abs=1e-6)


def test_ssmod_output_lengths_match_the_input():
    rng = np.random.default_rng(3163)
    r = ss(rng.standard_normal(123))
    for k in ("filtered_state", "smoothed_state", "filtered_state_variance"):
        assert np.asarray(r[k]).size == 123
