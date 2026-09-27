"""Hedderich methods (batch 3) against R: cor.test, chisq.test, poisson.test, qgamma, qbeta, pbinom."""

import math

from morie.fn.bnappx import binomial_normal_approx
from morie.fn.bvncnd import bivariate_normal_conditional
from morie.fn.chin1 import chi_square_n_minus_1
from morie.fn.corcin import correlation_ci_sample_size
from morie.fn.corrci import correlation_ci
from morie.fn.corrng import correlation_admissible_range
from morie.fn.ctresid import contingency_residuals
from morie.fn.gmconj import gamma_conjugate_posterior
from morie.fn.invbpr import inverse_binomial_prevalence
from morie.fn.irr import rate_ratio
from morie.fn.rrexct import rate_ratio_exact_ci


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_correlation_ci():
    r = correlation_ci(0.687, 50)
    # book p. 416: 0.5052731, 0.8103824
    assert close(r["lower"], 0.50527306379043613) and close(r["upper"], 0.81038243291053280)
    r = correlation_ci(0.90303030303030296, 10)
    assert close(r["lower"], 0.63371818108304401) and close(r["upper"], 0.97710335648264146)  # cor.test
    h = correlation_ci(0.687, 50, method="hotelling")
    z = math.atanh(0.687)
    assert close(h["z"], z - (3 * z + 0.687) / 200) and close(h["z"], 0.82618316681286408)
    assert close(h["lower"], 0.49765798223771052) and close(h["upper"], 0.80270722653182658)


def test_correlation_ci_sample_size():
    r = correlation_ci_sample_size(0.5, 0.8)
    assert close(r["n_exact"], 53.924556983524468) and r["n"] == 54  # book: 53.9 ~ 54


def test_admissible_range():
    r = correlation_admissible_range(0.6, 0.9)
    assert close(r["lower"], 0.19128808451674617) and close(r["upper"], 0.88871191548325390)


def test_chi_square_n_minus_1():
    r = chi_square_n_minus_1([[1, 5], [5, 1]])
    assert close(r["statistic"], 44 / 9) and close(r["p_value"], 0.027030076547772494)  # book 4.89
    assert close(chi_square_n_minus_1([[1, 5], [4, 2]])["statistic"], 2.828571428571428736)  # book 2.83


def test_contingency_residuals():
    r = contingency_residuals([[14, 22, 32], [18, 16, 8], [8, 2, 0]])
    pearson = [
        -1.82036410923641290,
        -0.14002800840280122,
        1.96039211763921339,
        1.06904496764969759,
        0.53452248382484879,
        -1.60356745147454638,
        2.55603860169077457,
        -0.73029674334022154,
        -1.82574185835055380,
    ]
    stdres = [
        -3.38682568717289012,
        -0.26052505285945354,
        3.64735074003234239,
        1.62399588588230026,
        0.81199794294115013,
        -2.43599382882345061,
        3.26969556547839568,
        -0.93419873299382761,
        -2.33549683248456885,
    ]
    for k in range(9):  # chisq.test residuals / stdres, by row
        i, j = k // 3, k % 3
        assert close(r["pearson"][i][j], pearson[k]) and close(r["adjusted"][i][j], stdres[k])
    assert close(r["statistic"], 21.576470588235292) and r["df"] == 4


def test_rate_ratio_exact():
    r = rate_ratio_exact_ci(40, 20, 22, 30)
    # book p. 370 and poisson.test: [1.5824, 4.8181]
    assert close(r["lower"], 1.58240228979166319689) and close(r["upper"], 4.81806982255807181303)
    assert close(r["ratio"], 2.72727272727272751496) and close(r["p_value"], 0.00012848004739237414, 1e-9)
    r = rate_ratio_exact_ci(3, 7.5, 11, 12, conf_level=0.9)
    assert close(r["lower"], 0.10412716066474971) and close(r["upper"], 1.39432918328082933)
    assert close(r["p_value"], 0.27337993090241935, 1e-9)


def test_gamma_conjugate():
    r = gamma_conjugate_posterior([0.8, 1.9, 0.4, 2.7, 1.1], 2, 1)
    assert r["shape"] == 7 and close(r["rate"], 7.9)
    assert close(r["lower"], 0.35624848753416022, 1e-10) and close(r["upper"], 1.65309797753401067, 1e-10)
    r = gamma_conjugate_posterior([3, 0, 2, 5], 2, 1, likelihood="poisson")
    assert r["shape"] == 12 and r["rate"] == 5
    assert close(r["lower"], 1.2401150217444432, 1e-10) and close(r["upper"], 3.9364077026603912, 1e-10)


def test_inverse_binomial_prevalence():
    r = inverse_binomial_prevalence(20, 100)
    # book p. 353: 0.16, [0.105, 0.238]
    assert close(r["estimate"], 19 / 119)
    assert close(r["lower"], 0.10487850197666279, 1e-10) and close(r["upper"], 0.23805309050431389, 1e-10)


def test_binomial_normal_approx():
    r = binomial_normal_approx(3, 10, 0.4)
    assert close(r["z"], -0.5 / math.sqrt(2.4)) and close(r["approx"], 0.37344281669518198)
    assert close(r["exact"], 0.38228060159999988)
    assert close(binomial_normal_approx(3, 10, 0.4, correct=False)["z"], -1 / math.sqrt(2.4))


def test_bivariate_normal_conditional():
    r = bivariate_normal_conditional(80, 170, 70, 10, 12, 0.6)
    assert close(r["mean"], 170 + 0.6 * 10 * 10 / 12) and close(r["sd"], 8.0)
    r = bivariate_normal_conditional(160, 170, 70, 10, 12, 0.6, given="x")
    assert close(r["mean"], 70 - 0.6 * 12) and close(r["sd"], 9.6)


def test_irr_zero_count():
    # rate_ratio in the R arm and morie.effect_sizes: 1/2 added to both counts when one is zero
    r = rate_ratio(0, 100, 12, 150)
    assert close(r.estimate, 0.06) and close(r.se, 1.4422205101855958276)
    assert close(r.ci_lower, 0.0035524741503428344) and close(r.ci_upper, 1.0133782394032559981)
