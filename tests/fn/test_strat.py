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


def test_fpc_variance_recomputed():
    d = {"y": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 14.0, 5.0], "stratum": ["a", "a", "a", "b", "b", "b", "b", "c"]}
    d["stratum"][-1] = "a"
    r = stratified_mean(d, pop_sizes={"a": 30, "b": 10})
    a, b = [1.0, 2.0, 3.0, 5.0], [10.0, 11.0, 12.0, 14.0]
    ma, mb = sum(a) / 4, sum(b) / 4
    sa = sum((v - ma) ** 2 for v in a) / 3
    sb = sum((v - mb) ** 2 for v in b) / 3
    var = 0.75**2 * (1 - 4 / 30) * sa / 4 + 0.25**2 * (1 - 4 / 10) * sb / 4
    assert abs(r.value - (0.75 * ma + 0.25 * mb)) < 1e-12
    assert abs(r.extra["se"] - var**0.5) < 1e-12
    assert abs(r.extra["ci_upper"] - r.value - 1.959963984540054 * r.extra["se"]) < 1e-12


def test_proportional_weights_without_fpc():
    d = {"y": [1.0, 2.0, 4.0, 10.0, 13.0], "stratum": [1, 1, 1, 2, 2]}
    r = stratified_mean(d)
    s1 = sum((v - 7 / 3) ** 2 for v in (1.0, 2.0, 4.0)) / 2
    s2 = 4.5
    assert abs(r.value - 30 / 5) < 1e-12
    assert abs(r.extra["se"] ** 2 - (0.36 * s1 / 3 + 0.16 * s2 / 2)) < 1e-12
