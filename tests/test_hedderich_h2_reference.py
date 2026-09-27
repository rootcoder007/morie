"""Hedderich methods (batch 2) against R: DescTools::GTest, prop.trend.test, MASS::fitdistr, nortest."""

import math

from morie.fn._stats_core import norm
from morie.fn.catrnd import cochran_armitage_test
from morie.fn.gamfit import gamma_fit
from morie.fn.lrgtst import g_test
from morie.fn.ntrtau import noether_tau_sample_size
from morie.fn.pchnrm import pearson_normality_test


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_g_test():
    # book p. 500: 2 (x log(x / n p0) + (n - x) log(...)) = 5.362487
    assert close(g_test([4, 56], [1 / 6, 5 / 6])["statistic"], 5.3624868993911265)
    r = g_test([18, 55, 27], [0.207025, 0.49595, 0.297025], n_estimated=1)
    # DescTools::GTest(o, p = e / 100): G = 1.1917 (the book prints 1.92)
    assert close(r["statistic"], 1.1916754720499076) and r["df"] == 1
    r = g_test([[14, 22, 32], [18, 16, 8], [8, 2, 0]])
    assert close(r["statistic"], 23.595462988817047) and close(r["p_value"], 9.6259434753331874e-05, 1e-10)


def test_cochran_armitage():
    r = cochran_armitage_test([22, 16, 2], [36, 34, 10], [1, 0, -1])
    # prop.trend.test / prop.test (book p. 730: 5.220 and 5.495)
    assert close(r["chi2_trend"], 5.2197070572569864) and close(r["chi2_total"], 5.4954248366013072)
    zu = (22 / 36 - 0.2) / math.sqrt((22 / 36) * (14 / 36) / 36 + 0.2 * 0.8 / 10)
    assert close(r["z_unpooled"], zu)


def test_noether():
    assert close(noether_tau_sample_size(0.3)["n_exact"], 38.759899922711519)
    assert noether_tau_sample_size(0.3)["n"] == 39
    assert close(noether_tau_sample_size(0.5, 0.01, 0.9)["n_exact"], 26.452243856441832)


def test_gamma_fit():
    tm = [
        274.0,
        1.7,
        871.0,
        1311.0,
        236.0,
        458.0,
        54.9,
        1787.0,
        0.75,
        776.0,
        28.5,
        20.8,
        363.0,
        1661.0,
        828.0,
        290.0,
        175.0,
        970.0,
        1278.0,
        126.0,
    ]
    r = gamma_fit(tm, "moments")
    # book p. 297: k = 1.0559, lambda = 0.0018
    assert close(r["shape"], 1.0559039471895129) and close(r["rate"], 0.0018346556401063587)
    r = gamma_fit(tm)
    # MASS::fitdistr(tm, "gamma"), shape equation solved exactly
    assert close(r["shape"], 0.57918183671663281, 1e-11) and close(r["rate"], 0.0010063408004181045, 1e-11)


def test_pearson_normality_matches_nortest():
    x = [float(norm.ppf((i - 0.5) / 40)) * 2 + 5 + 0.3 * math.sin(i) for i in range(1, 41)]
    r = pearson_normality_test(x)
    assert close(r["statistic"], 2.3) and close(r["p_value"], 0.89014511791686335, 1e-10) and r["n_classes"] == 9
    r = pearson_normality_test(x, n_classes=8)
    assert close(r["statistic"], 1.2) and close(r["p_value"], 0.94487736500212194, 1e-10)
