from morie.effects import estimate_ate
from morie.fn import _frame_core as pd


def test_estimate_ate_returns_float_estimate_and_standard_error():
    data = pd.DataFrame(
        {
            "outcome": [1.0, 3.0, 2.0, 4.0, 1.5, 4.5],
            "treated": [0, 1, 0, 1, 0, 1],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    ate, se = estimate_ate(
        data=data,
        outcome="outcome",
        treatment="treated",
        weights_col="weight",
    )

    assert isinstance(ate, float)
    assert isinstance(se, float)
    assert ate > 0


def test_hedges_exact_j_and_hedges_olkin_se():
    # effectsize::hedges_g / cohens_d and metafor::escalc("SMD") on these data
    import math

    from morie.effect_sizes import cles, cohens_d, hedges_g

    x = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4]
    y = [4.2, 5.0, 3.9, 5.8, 4.4, 4.9, 5.3, 4.1, 4.6]
    d = cohens_d(x, y)
    assert abs(d.estimate - 1.6559198522887781) <= 1e-12
    assert abs(d.se - math.sqrt(16 / 63 + d.estimate**2 / 32)) <= 1e-14
    m = 14
    j = math.exp(math.lgamma(m / 2) - 0.5 * math.log(m / 2) - math.lgamma((m - 1) / 2))
    g = hedges_g(x, y)
    assert abs(g.estimate - 1.5653207170841281) <= 1e-12
    assert abs(g.extra["correction_factor"] - j) <= 1e-15
    # pairs: x > y in 5 of 9, ties in 2, counted as 1/2 each
    assert cles([1, 2, 3], [2, 2, 0]).estimate == (5 + 0.5 * 2) / 9


def test_zero_cells_metafor_correction_and_nnt_span():
    # metafor::escalc("OR"/"RR"/"IRR") values
    import math

    from morie.effect_sizes import (
        number_needed_to_treat,
        odds_ratio,
        r_effect_size,
        r_squared,
        rate_ratio,
        risk_difference,
        risk_ratio,
    )

    o = odds_ratio(7, 0, 4, 10)
    assert abs(o.estimate - (7.5 * 10.5) / (0.5 * 4.5)) <= 1e-13
    assert abs(o.se - math.sqrt(1 / 7.5 + 1 / 0.5 + 1 / 4.5 + 1 / 10.5)) <= 1e-14
    assert abs(odds_ratio(12, 5, 3, 9).se - 0.85309892613798188) <= 1e-12
    assert abs(risk_ratio(7, 0, 4, 10).estimate - 3.125) <= 1e-13
    assert abs(risk_ratio(7, 0, 4, 10).se - 0.40483192671637058) <= 1e-12
    assert abs(rate_ratio(0, 120, 6, 150).estimate - 0.096153846153846145) <= 1e-14
    assert abs(rate_ratio(0, 120, 6, 150).se - 1.4675987714106855) <= 1e-12
    rd = risk_difference(5, 5, 4, 6)
    nn = number_needed_to_treat(5, 5, 4, 6)
    assert rd.ci_lower < 0 < rd.ci_upper
    assert abs(nn.ci_lower - 1 / rd.ci_upper) <= 1e-14
    assert nn.ci_upper == math.inf and nn.extra["ci_spans_zero"]
    x, y = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 2]
    r, r2 = r_effect_size(x, y), r_squared(x, y)
    assert r.ci_lower < 0 < r.ci_upper
    assert r2.ci_lower == 0.0 and r2.ci_upper == max(r.ci_lower**2, r.ci_upper**2)


def test_rank_biserial_sign_and_bergsma_v():
    # effectsize::rank_biserial and effectsize::cramers_v(adjust = TRUE)
    import math

    from morie.effect_sizes import cliffs_delta, cramers_v, rank_biserial_correlation

    x = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4]
    y = [4.2, 5.0, 3.9, 5.8, 4.4, 4.9, 5.3, 4.1, 4.6]
    rb = rank_biserial_correlation(x, y).estimate
    assert abs(rb - 0.7777777777777778) <= 1e-14
    assert abs(rb - cliffs_delta(x, y).estimate) <= 1e-14
    tb = [[12, 3, 8], [5, 9, 4], [7, 6, 10]]
    cv = cramers_v(tb)
    assert abs(cv.estimate - 0.2556545) <= 1e-6
    assert abs(cv.extra["bias_corrected_v"] - 0.1863203) <= 1e-6
    n = 64
    chi2 = n * 2 * cv.estimate**2
    assert abs(cv.extra["bias_corrected_v"] - math.sqrt((chi2 / n - 4 / (n - 1)) / (3 - 4 / (n - 1) - 1))) <= 1e-14
