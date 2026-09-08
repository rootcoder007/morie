"""slvgrf -- TOC/RATE/Qini. Source: Yadlowsky et al. (2025) JASA
120(549), 38-51; Sverdrup et al. (2025) JCGS 34(3), 948-960."""
import pytest

from morie.fn.slvgrf import (aipw_scores, autoc, qini_coefficient,
                             qini_curve, rate, rate_test, toc_curve)


def test_toc_ends_at_zero_by_construction():
    g = [3.0, 1.0, -1.0, 2.0, 0.5]
    c = toc_curve(g, [5.0, 4.0, 3.0, 2.0, 1.0])
    assert c["toc"][-1] == pytest.approx(0.0, abs=1e-13)


def test_toc_first_point_is_the_top_unit_minus_the_ate():
    g = [3.0, 1.0, -1.0, 2.0, 0.5]
    c = toc_curve(g, [5.0, 4.0, 3.0, 2.0, 1.0])
    assert c["toc"][0] == pytest.approx(3.0 - sum(g) / 5, abs=1e-13)


def test_a_perfect_ranking_gives_a_positive_autoc():
    g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert autoc(g, list(range(6))) > 0.0


def test_a_reversed_ranking_gives_the_mirror_sign():
    g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    up = autoc(g, list(range(6)))
    down = autoc(g, list(reversed(range(6))))
    assert up > 0.0 > down


def test_qini_and_autoc_are_the_same_metric_at_different_weights():
    g = [1.0, 5.0, 2.0, 4.0]
    p = [4.0, 3.0, 2.0, 1.0]
    assert qini_coefficient(g, p) == pytest.approx(
        rate(g, p, weight="qini")["estimate"], abs=1e-15)
    assert autoc(g, p) == pytest.approx(
        rate(g, p, weight="autoc")["estimate"], abs=1e-15)


def test_qini_curve_ends_at_the_ate():
    g = [1.0, 5.0, 2.0, 4.0]
    c = qini_curve(g, [4.0, 3.0, 2.0, 1.0])
    assert c["gain"][-1] == pytest.approx(c["ate"], abs=1e-13)


def test_aipw_score_is_the_textbook_expression():
    s = aipw_scores([10.0], [1.0], [4.0], [1.0], 0.5)
    assert s[0] == pytest.approx(4.0 - 1.0 + (10.0 - 4.0) / 0.5)


def test_aipw_reduces_to_the_outcome_difference_with_no_residual():
    s = aipw_scores([4.0], [1.0], [4.0], [1.0], 0.5)
    assert s[0] == pytest.approx(3.0, abs=1e-13)


def test_rate_test_reports_a_finite_standard_error():
    g = [float(i) for i in range(40)]
    r = rate_test(g, list(range(40)), reps=40, seed=1)
    assert r["se"] > 0.0
    assert r["p_value"] <= 1.0


def test_extreme_propensity_is_refused():
    with pytest.raises(ValueError):
        aipw_scores([1.0], [1.0], [0.0], [0.0], 1.0)


def test_non_binary_treatment_is_refused():
    with pytest.raises(ValueError):
        aipw_scores([1.0], [0.3], [0.0], [0.0], 0.5)


def test_unknown_weight_is_refused():
    with pytest.raises(ValueError):
        rate([1.0, 2.0], [1.0, 2.0], weight="cubic")


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        toc_curve([1.0, 2.0], [1.0])


def test_too_few_units_for_the_half_sample_bootstrap_is_refused():
    with pytest.raises(ValueError):
        rate_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_negative_cost_is_refused():
    with pytest.raises(ValueError):
        qini_curve([1.0, 2.0], [1.0, 2.0], cost=[1.0, -1.0])
