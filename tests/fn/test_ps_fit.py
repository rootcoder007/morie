"""Tests for morie.fn.ps_fit — propensity score estimation via logistic regression."""

import math
from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import pytest

from morie.fn.ps_fit import compute_propensity_scores


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logits = [0.5 * x1[i] - 0.3 * x2[i] for i in range(n)]
    probs = [1.0 / (1.0 + math.exp(-l)) for l in logits]
    u = rng.uniform(0, 1, n)
    t = [1 if u[i] < probs[i] else 0 for i in range(n)]
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t})


def test_returns_series(synth_data):
    ps = compute_propensity_scores(synth_data, treatment="treatment", covariates=["x1", "x2"])
    assert hasattr(ps, "index") and hasattr(ps, "tolist")


def test_values_in_unit_interval(synth_data):
    ps = compute_propensity_scores(synth_data, treatment="treatment", covariates=["x1", "x2"])
    vals = ps.tolist()
    assert all(0 < v < 1 for v in vals)


def test_length_matches_input(synth_data):
    ps = compute_propensity_scores(synth_data, treatment="treatment", covariates=["x1", "x2"])
    assert len(ps) == len(synth_data)


def test_works_with_categorical_covariate():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "treatment": rng.integers(0, 2, n),
        }
    )
    ps = compute_propensity_scores(df, treatment="treatment", covariates=["x1", "cat"])
    assert len(ps) == n
    vals = ps.tolist()
    assert all(0 < v < 1 for v in vals)


def test_index_preserved(synth_data):
    df = synth_data.copy()
    df.index = range(100, 100 + len(df))
    ps = compute_propensity_scores(df, treatment="treatment", covariates=["x1", "x2"])
    assert list(ps.index) == list(df.index)
