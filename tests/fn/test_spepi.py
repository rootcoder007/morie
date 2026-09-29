"""Tests for spepi: spatial epidemiology."""

import math

from morie.fn.spepi import (
    _rgamma,
    areal_wombling,
    buffer_rate_ratio,
    funnel_control_limits,
    ghose_drug_filter,
    kernel_relative_risk,
    need_based_allocation,
    oden_ipop,
    poisson_ecological,
    prospective_scan,
    zip_gibbs,
)


def test_marsaglia_tsang_moments():
    for shape in (0.5, 2.5, 7.0):
        g = [_rgamma(shape, 11, k) for k in range(3000)]
        m = sum(g) / 3000
        assert abs(m - shape) <= 4 * math.sqrt(shape / 3000)


def test_zip_gibbs_finds_structural_zeros():
    y = [0] * 12 + [3, 4, 5, 2, 6, 3, 4, 5]
    E = [1.0] * 20
    r = zip_gibbs(y, E, 1500, burn=200, seed=5)
    # regions with positive counts are never structural zeros: theta | y ~ Gamma(1 + y, 1 + E)
    for i in range(12, 20):
        want = (1 + y[i]) / 2.0
        assert abs(r.theta[i] - want) <= 4 * math.sqrt(want / 2.0 / 1300) * 3
    assert 0.0 < r.pi < 0.6


def test_kernel_relative_risk_symmetry():
    cs = [(0.0, 0.0), (1.0, 0.0)]
    ct = [(0.0, 1.0), (1.0, 1.0)]
    r = kernel_relative_risk(cs, ct, [(0.5, 0.5), (0.5, 0.0)], 0.4)
    assert abs(r.log_rr[0]) <= 1e-12 and r.log_rr[1] > 0


def test_oden_zero_when_cases_follow_population():
    W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    r = oden_ipop([10, 20, 30, 40], [100, 200, 300, 400], W)
    assert abs(r.statistic) <= 1e-12


def test_prospective_scan_finds_the_recent_cluster():
    cnt = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 9]]
    ex = [[1.0] * 4 for _ in range(3)]
    r = prospective_scan(cnt, ex, [(0, 0), (1, 0), (2, 0), (9, 0)], max_window=2, nsim=19, seed=1)
    assert r.cluster == [3] and r.window == 1 and r.p_value <= 0.1


def test_poisson_ecological_exact_fit():
    r = poisson_ecological([2, 4, 8], [1.0, 1.0, 1.0], [[0.0], [1.0], [2.0]])
    assert abs(r.beta[1] - math.log(2)) <= 1e-12 and abs(r.deviance) <= 1e-10


def test_wombling_buffer_allocation_funnel_ghose():
    w = areal_wombling([0.0, 0.1, 5.0, 5.2], [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], quantile=0.9)
    assert w.barriers == [[1, 2]]
    b = buffer_rate_ratio([6, 2, 2], [100, 100, 100], [(0, 0), (5, 0), (9, 0)], [(0, 0)], 1.0)
    want = sum(math.comb(10, k) * (1 / 3) ** k * (2 / 3) ** (10 - k) for k in range(6, 11))
    assert abs(b.p_value - want) <= 1e-12
    a = need_based_allocation(100.0, [1, 1], [1.0, 3.0], floor_share=0.5)
    assert a == [37.5, 62.5]
    f = funnel_control_limits(0.2, [100], z=(3.0,))
    assert abs(f.upper[0][0] - (0.2 + 3 * 0.04)) <= 1e-15
    assert ghose_drug_filter([300], [-0.5], [80], [30]) == [False]
