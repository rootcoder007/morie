"""kalmn: linear Kalman filter predict-update recursion (Kalman 1960)."""

import numpy as np
import pytest

from morie.fn.kalmn import kalman_filter as kf


def test_kalmn_noiseless_observation_locks_onto_the_measurement():
    """With R = 0 the filter must trust the observation completely."""
    y = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    r = kf(y, F=np.eye(1), H=np.eye(1), Q=np.eye(1) * 1.0, R=np.zeros((1, 1)))
    assert np.asarray(r["state"]).ravel() == pytest.approx(y.ravel())


def test_kalmn_huge_observation_noise_ignores_the_data():
    """With R enormous and Q = 0 the state should barely move from x0."""
    y = np.full((30, 1), 100.0)
    r = kf(y, F=np.eye(1), H=np.eye(1), Q=np.zeros((1, 1)),
           R=np.eye(1) * 1e12, x0=np.zeros(1), P0=np.eye(1) * 1e-6)
    assert abs(float(np.asarray(r["state"]).ravel()[-1])) < 1.0


def test_kalmn_tracks_a_constant_signal_under_noise():
    rng = np.random.default_rng(2501)
    truth = 5.0
    y = (truth + rng.normal(0, 1.0, 400)).reshape(-1, 1)
    r = kf(y, F=np.eye(1), H=np.eye(1), Q=np.eye(1) * 1e-6, R=np.eye(1))
    assert float(np.asarray(r["state"]).ravel()[-1]) == pytest.approx(truth, abs=0.3)


def test_kalmn_state_covariance_shrinks_as_evidence_accumulates():
    """P must decrease monotonically for a static state with no process
    noise -- more data can only reduce uncertainty."""
    rng = np.random.default_rng(2503)
    y = (2.0 + rng.normal(0, 1.0, 60)).reshape(-1, 1)
    r = kf(y, F=np.eye(1), H=np.eye(1), Q=np.zeros((1, 1)), R=np.eye(1))
    P = np.asarray(r["state_cov"]).reshape(60, -1)[:, 0]
    assert np.all(np.diff(P) <= 1e-12)


def test_kalmn_innovations_are_white_for_a_correctly_specified_model():
    """The standard filter diagnostic: if the model is right the one-step
    prediction errors carry no remaining autocorrelation."""
    rng = np.random.default_rng(2509)
    y = (3.0 + rng.normal(0, 1.0, 2000)).reshape(-1, 1)
    v = np.asarray(kf(y, F=np.eye(1), H=np.eye(1),
                      Q=np.zeros((1, 1)), R=np.eye(1))["innovations"]).ravel()
    v = v[50:]
    ac1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    assert abs(ac1) < 0.1


def test_kalmn_reports_a_finite_loglikelihood():
    rng = np.random.default_rng(2521)
    y = rng.normal(0, 1, 100).reshape(-1, 1)
    r = kf(y, F=np.eye(1), H=np.eye(1), Q=np.eye(1) * 0.1, R=np.eye(1))
    assert np.isfinite(r["loglik"])
    assert r["n"] == 100


def test_kalmn_loglik_prefers_the_true_observation_variance():
    """Scanning R, the likelihood should peak near the variance that actually
    generated the data -- the property that makes the filter usable for
    estimation, not just smoothing."""
    rng = np.random.default_rng(2531)
    y = rng.normal(0, 2.0, 1500).reshape(-1, 1)
    grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    lls = [kf(y, F=np.eye(1), H=np.eye(1), Q=np.zeros((1, 1)),
              R=np.eye(1) * g)["loglik"] for g in grid]
    assert grid[int(np.argmax(lls))] == pytest.approx(4.0)  # variance = 2^2
