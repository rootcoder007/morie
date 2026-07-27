"""DiD cluster: causdid2, drovrl, drbqs."""

import numpy as np
import pytest

from morie.fn.causdid2 import causal_did_2x2
from morie.fn.drbqs import dr_did_quantile
from morie.fn.drovrl import dr_did_overlap_trim


def test_causdid2_hand():
    # cell means: T0=1, T1=4, C0=2, C1=3 -> ATT = 3 - 1 = 2
    y = np.array([1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 3.0, 3.0])
    T = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    P = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    out = causal_did_2x2(y, T, P)
    assert out["att"] == pytest.approx(2.0)
    assert out["cell_means"] == {"T0": 1.0, "T1": 4.0, "C0": 2.0, "C1": 3.0}
    with pytest.raises(ValueError):
        causal_did_2x2(y[:4], T[:4], P[:4])  # missing control cells


def test_causdid2_recovery_with_group_trends():
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 800
        T = (rng.random(n) < 0.5).astype(float)
        P = (rng.random(n) < 0.5).astype(float)
        # group gap 1.5, common trend 0.8, effect 2 in the treated-post cell
        y = 1.5 * T + 0.8 * P + 2.0 * T * P + rng.normal(scale=1.0, size=n)
        out = causal_did_2x2(y, T, P)
        hits += abs(out["att"] - 2.0) < 3 * out["se"]
    assert hits >= 7  # measured 8/8


def test_drovrl_trims_and_recovers():
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 2000
        x = rng.normal(size=n)
        e = 1 / (1 + np.exp(-2.0 * x))
        D = (rng.random(n) < e).astype(float)
        fe = rng.normal(size=n)
        y_pre = fe + 0.5 * x + rng.normal(scale=0.5, size=n)
        # trend depends on x (confounded), effect 1.5 on the treated
        y_post = fe + 0.5 * x + 0.8 * x + 1.5 * D + rng.normal(scale=0.5, size=n)
        out = dr_did_overlap_trim(y_pre, y_post, D, x, eps=0.1)
        assert out["n_trimmed"] > 0  # 2.0*x propensity has mass outside [.1,.9]
        assert out["n_kept"] + out["n_trimmed"] == n
        hits += abs(out["att"] - 1.5) < 0.25
    assert hits >= 7  # measured 8/8
    with pytest.raises(ValueError):
        dr_did_overlap_trim(y_pre, y_post, D, x, eps=0.7)


def test_drbqs_constant_shift():
    # effect 2 uniformly on the change distribution -> QTT(tau) = 2 at all tau
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 2500
        x = rng.normal(size=n)
        D = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
        y_pre = 0.5 * x + rng.normal(size=n)
        y_post = y_pre + 0.6 * x + 2.0 * D + rng.normal(scale=0.3, size=n)
        out = dr_did_quantile(y_pre, y_post, D, x, quantile=[0.25, 0.5, 0.75])
        hits += np.all(np.abs(out["qtt"] - 2.0) < 0.35)
    assert hits >= 7  # measured 8/8


def test_drbqs_validation():
    with pytest.raises(ValueError):
        dr_did_quantile([1.0, 2.0], [1.0, 2.0], [1, 0], [0.0, 1.0], quantile=1.2)
    with pytest.raises(ValueError):
        dr_did_quantile([1.0, 2.0], [1.0, 2.0], [1, 1], [0.0, 1.0])
