"""didfst -- DiD with a forest. Source: Wager (2025) ch. 13;
Callaway & Sant'Anna (2021) JoE 225(2), 200-230."""
import pytest

from morie.fn.didfst import (aggregate_att, did_estimate, did_forest,
                             group_time_att, panel_differences,
                             placebo_did)


def panel():
    # 4 units, 4 periods, H = 2. Treated units gain exactly 3.
    return ([[1.0, 2.0, 6.0, 7.0],
             [2.0, 3.0, 7.0, 8.0],
             [0.0, 1.0, 2.0, 3.0],
             [5.0, 6.0, 7.0, 8.0]], [1.0, 1.0, 0.0, 0.0])


def test_panel_differences_is_post_mean_minus_pre_mean():
    Y, _ = panel()
    d = panel_differences(Y, 2)
    assert d[0] == pytest.approx((6.0 + 7.0) / 2 - (1.0 + 2.0) / 2)
    assert d[2] == pytest.approx((2.0 + 3.0) / 2 - (0.0 + 1.0) / 2)


def test_did_recovers_the_planted_effect_exactly():
    Y, D = panel()
    est, _, _, _, _ = did_estimate(panel_differences(Y, 2), D)
    assert est == pytest.approx(3.0, abs=1e-12)


def test_uniform_weights_reproduce_the_scalar_estimator():
    Y, D = panel()
    d = panel_differences(Y, 2)
    flat, _, _, _, _ = did_estimate(d, D)
    w, _, _, _, _ = did_estimate(d, D, weights=[0.25] * 4)
    assert w == pytest.approx(flat, abs=1e-13)


def test_estimator_is_scale_invariant_in_the_weights():
    Y, D = panel()
    d = panel_differences(Y, 2)
    a, _, _, _, _ = did_estimate(d, D, weights=[1.0] * 4)
    b, _, _, _, _ = did_estimate(d, D, weights=[9.0] * 4)
    assert a == pytest.approx(b, abs=1e-12)


def test_forest_returns_one_estimate_per_row():
    # the forest subsamples, so it needs more than a handful of rows
    Y, D, X = [], [], []
    for i in range(60):
        d = 1.0 if i % 2 == 0 else 0.0
        base = float(i) / 10.0
        Y.append([base, base + 1.0, base + 2.0 + 3.0 * d,
                  base + 3.0 + 3.0 * d])
        D.append(d)
        X.append([float(i % 7) / 7.0])
    r = did_forest(Y, D, X, 2, n_trees=10, min_leaf=5, seed=0)
    assert len(r["tau"]) == 60
    assert r["att_uniform"] == pytest.approx(3.0, abs=1e-12)


def test_placebo_is_zero_when_pre_trends_are_parallel():
    Y = [[1.0, 2.0, 3.0, 9.0], [2.0, 3.0, 4.0, 10.0],
         [0.0, 1.0, 2.0, 3.0], [5.0, 6.0, 7.0, 8.0]]
    r = placebo_did(Y, [1.0, 1.0, 0.0, 0.0], 3, split=1)
    assert r["estimate"] == pytest.approx(0.0, abs=1e-12)


def test_placebo_detects_a_planted_pre_trend():
    Y = [[1.0, 3.0, 5.0, 9.0], [2.0, 4.0, 6.0, 10.0],
         [0.0, 1.0, 2.0, 3.0], [5.0, 6.0, 7.0, 8.0]]
    r = placebo_did(Y, [1.0, 1.0, 0.0, 0.0], 3, split=1)
    assert abs(r["estimate"]) > 0.5


def test_group_time_att_is_the_double_difference():
    Y = [[0.0, 1.0, 5.0, 6.0],      # cohort g = 3
         [1.0, 2.0, 6.0, 7.0],
         [0.0, 1.0, 2.0, 3.0],      # never treated
         [2.0, 3.0, 4.0, 5.0]]
    gt = group_time_att(Y, [3, 3, None, None])
    assert gt["att"][(3, 3)]["att"] == pytest.approx(3.0, abs=1e-12)
    assert gt["att"][(3, 4)]["att"] == pytest.approx(3.0, abs=1e-12)


def test_not_yet_treated_is_the_larger_comparison_group():
    Y = [[0.0, 1.0, 5.0, 6.0], [0.0, 1.0, 2.0, 7.0],
         [0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]]
    a = group_time_att(Y, [3, 4, None, None],
                       comparison="never-treated")
    b = group_time_att(Y, [3, 4, None, None],
                       comparison="not-yet-treated")
    assert (b["att"][(3, 3)]["n_control"]
            > a["att"][(3, 3)]["n_control"])


def test_aggregate_event_profile_keys_are_event_times():
    Y = [[0.0, 1.0, 5.0, 6.0], [1.0, 2.0, 6.0, 7.0],
         [0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]]
    gt = group_time_att(Y, [3, 3, None, None])
    prof = aggregate_att(gt, scheme="event")["profile"]
    assert sorted(prof) == [0, 1]


def test_unbalanced_panel_is_refused():
    with pytest.raises(ValueError):
        panel_differences([[1.0, 2.0], [3.0]], 1)


def test_event_time_out_of_range_is_refused():
    Y, _ = panel()
    with pytest.raises(ValueError):
        panel_differences(Y, 4)


def test_missing_comparison_group_is_refused():
    Y, _ = panel()
    with pytest.raises(ValueError):
        did_estimate(panel_differences(Y, 2), [1.0] * 4)


def test_treatment_in_period_one_is_refused():
    Y, _ = panel()
    with pytest.raises(ValueError):
        group_time_att(Y, [1, 1, None, None])


def test_non_binary_adoption_is_refused():
    Y, _ = panel()
    with pytest.raises(ValueError):
        did_estimate(panel_differences(Y, 2), [0.5, 1.0, 0.0, 0.0])
