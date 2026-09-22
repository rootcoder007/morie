# SPDX-License-Identifier: AGPL-3.0-or-later
"""estimate_double_ml() and estimate_irm() return a plain dict with ate/se.

They returned a DoubleMLPLR-like object with .coef/.se; the change was
unrecorded and the canonical tests that would have caught it were muted.
This is the live contract test (WHATS_NEW 1.3.2).
"""
import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd


def _dgp(n=600, tau=2.0, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n).tolist()
    x2 = rng.normal(0, 1, n).tolist()
    t = [1 if 0.6 * a + rng.normal(0, 1) > 0 else 0 for a in x1]
    y = [tau * ti + a + 0.5 * b + rng.normal(0, 1) for ti, a, b in zip(t, x1, x2)]
    return pd.DataFrame({"outcome": y, "treatment": t, "X1": x1, "X2": x2})


def test_double_ml_returns_dict_with_ate_and_se():
    from morie.causal import estimate_double_ml

    out = estimate_double_ml(data=_dgp(), outcome="outcome", treatment="treatment",
                             covariates=["X1", "X2"], n_folds=3)
    assert isinstance(out, dict) and {"ate", "se"} <= set(out)
    assert math.isfinite(float(out["ate"])) and float(out["se"]) > 0
    assert abs(float(out["ate"]) - 2.0) < 0.6


def test_irm_returns_dict_with_ate_and_se():
    from morie.causal import estimate_irm

    out = estimate_irm(_dgp(tau=1.0), treatment="treatment", outcome="outcome",
                       covariates=["X1", "X2"], n_folds=3)
    assert isinstance(out, dict) and {"ate", "se"} <= set(out)
    assert math.isfinite(float(out["ate"])) and float(out["se"]) > 0
    assert abs(float(out["ate"]) - 1.0) < 0.6
