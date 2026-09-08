"""saigeg -- SAIGE. Source: Zhou, W. et al. (2018) Nature Genetics,
doi:10.1038/s41588-018-0184-y."""
import math

import pytest

from morie.fn import _array_core as np
from morie.fn.saigeg import (cgf, normal_pvalue, saddlepoint_pvalue,
                             saige_test, score_statistic,
                             variance_ratio)


def setup(n=1500, mu=0.03, seed=3):
    rng = np.random.default_rng(seed)
    MU = [mu] * n
    G = [1.0 if float(rng.uniform()) < 0.2 else 0.0 for _ in range(n)]
    return MU, G


def test_cgf_is_zero_at_the_origin():
    MU, G = setup()
    assert cgf(0.0, G, MU, 0) == pytest.approx(0.0, abs=1e-12)


def test_first_derivative_is_zero_at_the_origin():
    MU, G = setup()
    assert cgf(0.0, G, MU, 1) == pytest.approx(0.0, abs=1e-12)


def test_second_derivative_is_the_score_variance():
    MU, G = setup()
    want = sum(G[i] ** 2 * MU[i] * (1 - MU[i]) for i in range(len(G)))
    assert cgf(0.0, G, MU, 2) == pytest.approx(want, abs=1e-9)


def test_the_score_is_the_stated_sum():
    s = score_statistic([1.0] * 5 + [0.0] * 15, [1.0] * 20,
                        [0.25] * 20)
    assert s["score"] == pytest.approx(5 * 0.75 + 15 * -0.25,
                                       abs=1e-12)


def test_the_score_variance_is_the_stated_sum():
    s = score_statistic([1.0] * 5 + [0.0] * 15, [1.0] * 20,
                        [0.25] * 20)
    assert s["variance"] == pytest.approx(20 * 0.25 * 0.75,
                                          abs=1e-12)


def test_saddlepoint_solves_its_defining_equation():
    MU, G = setup()
    s = 3.0 * math.sqrt(cgf(0.0, G, MU, 2))
    r = saddlepoint_pvalue(s, G, MU)
    assert cgf(r["t_hat"], G, MU, 1) == pytest.approx(s, abs=1e-6)


def test_the_curvature_at_the_saddlepoint_is_positive():
    MU, G = setup()
    s = 3.0 * math.sqrt(cgf(0.0, G, MU, 2))
    assert saddlepoint_pvalue(s, G, MU)["K2"] > 0.0


def test_balanced_classes_agree_with_the_normal():
    _, G = setup()
    MUb = [0.5] * len(G)
    s = 0.5 * math.sqrt(cgf(0.0, G, MUb, 2))
    a = normal_pvalue(s, cgf(0.0, G, MUb, 2))["p_value"]
    b = saddlepoint_pvalue(s, G, MUb)["p_value"]
    assert b == pytest.approx(a, rel=0.05)


def test_imbalance_makes_the_normal_anti_conservative():
    MU, G = setup(mu=0.02)
    s = 5.0 * math.sqrt(cgf(0.0, G, MU, 2))
    a = normal_pvalue(s, cgf(0.0, G, MU, 2))["p_value"]
    b = saddlepoint_pvalue(s, G, MU)["p_value"]
    assert b > a


def test_both_p_values_are_proper_probabilities():
    MU, G = setup()
    for mult in (0.5, 2.0, 4.0):
        s = mult * math.sqrt(cgf(0.0, G, MU, 2))
        assert 0.0 <= saddlepoint_pvalue(s, G, MU)["p_value"] <= 1.0


def test_a_zero_score_gives_a_p_value_near_one():
    MU, G = setup()
    assert saddlepoint_pvalue(0.0, G, MU)["p_value"] == \
        pytest.approx(1.0, abs=1e-6)


def test_variance_ratio_of_a_doubled_series_is_four():
    r = variance_ratio([1.0, 2.0, 3.0], [0.5, 1.0, 1.5])
    assert r["ratio"] == pytest.approx(4.0, abs=1e-12)


def test_saige_test_reports_both_p_values():
    y = [1.0] * 30 + [0.0] * 270
    g = [1.0 if i % 4 == 0 else 0.0 for i in range(300)]
    r = saige_test(y, g)
    assert 0.0 <= r["p_value"] <= 1.0
    assert 0.0 <= r["p_normal"] <= 1.0
    assert r["n_cases"] == 30 and r["n_controls"] == 270


def test_a_monomorphic_variant_is_refused():
    with pytest.raises(ValueError):
        score_statistic([1.0, 0.0], [0.0, 0.0], [0.5, 0.5])


def test_a_single_class_phenotype_is_refused():
    with pytest.raises(ValueError):
        saige_test([0.0] * 20, [1.0] * 20)


def test_a_non_binary_phenotype_is_refused():
    with pytest.raises(ValueError):
        saige_test([0.3] * 20, [1.0] * 20)


def test_a_degenerate_fitted_mean_is_refused():
    with pytest.raises(ValueError):
        score_statistic([1.0, 0.0], [1.0, 1.0], [1.0, 0.5])


def test_an_unreachable_score_is_refused():
    MU, G = setup()
    with pytest.raises(ValueError):
        saddlepoint_pvalue(1e9, G, MU)


def test_an_invalid_cgf_order_is_refused():
    MU, G = setup()
    with pytest.raises(ValueError):
        cgf(0.0, G, MU, 3)


def test_a_non_positive_variance_is_refused():
    with pytest.raises(ValueError):
        normal_pvalue(1.0, 0.0)
