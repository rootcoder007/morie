"""Tests for morie.fn.strat — Stratified mean estimator."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.strat import strat, stratified_mean


def test_equal_strata():
    """Equal-sized strata: stratified mean = overall mean."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(200),
            "stratum": np.repeat([0, 1], 100),
        }
    )
    result = stratified_mean(df)
    overall = df["y"].mean()
    assert abs(result.value - overall) < 0.01


def test_weighted_mean():
    """With population sizes, the weighted mean should differ from simple mean."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "y": np.concatenate([rng.normal(10, 1, 50), rng.normal(20, 1, 50)]),
            "stratum": np.repeat([0, 1], 50),
        }
    )
    # A (stratum 0) is 90% of population
    result = stratified_mean(df, pop_sizes={0: 900, 1: 100})
    # Independent computation of the documented formula:
    # W_h = N_h / N; y_bar_st = sum(W_h * y_bar_h)
    N_total = 900 + 100
    y_bar_A = float(df["y"][df["stratum"] == 0].mean())
    y_bar_B = float(df["y"][df["stratum"] == 1].mean())
    expected = (900 / N_total) * y_bar_A + (100 / N_total) * y_bar_B
    assert abs(result.value - expected) < 1e-9
    assert result.value < 15  # Weighted toward A's mean of 10
    assert result.extra["n_strata"] == 2


def test_strat_alias():
    assert strat is stratified_mean
