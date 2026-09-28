import itertools
import math

from morie.fn.effectmod import submax_comparisons, submax_test, truncated_product_pvalue, wilcoxon_sensitivity_moments


def _zaykin(p, tau):
    L = len(p)
    w = math.prod(v for v in p if v <= tau)
    tot = 0.0
    for k in range(1, L + 1):
        if w <= tau**k:
            inner = w * sum((k * math.log(tau) - math.log(w)) ** s / math.factorial(s) for s in range(k))
        else:
            inner = tau**k
        tot += math.comb(L, k) * (1 - tau) ** (L - k) * inner
    return tot


def test_truncated_product_matches_zaykin():
    for p, tau in (([0.01, 0.03, 0.2, 0.04, 0.5], 0.05), ([0.001, 0.02, 0.09, 0.15], 0.2), ([0.025, 0.4], 0.05)):
        assert abs(truncated_product_pvalue(p, tau).pvalue - _zaykin(p, tau)) < 1e-12
    assert truncated_product_pvalue([0.3, 0.2], 0.1).pvalue == 1.0


def test_wilcoxon_moments_by_enumeration():
    n = 6
    vals = [sum(r for r, s in zip(range(1, n + 1), signs) if s) for signs in itertools.product((0, 1), repeat=n)]
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    m = wilcoxon_sensitivity_moments(n)
    assert abs(m.mu - mu) < 1e-12 and abs(m.nu - var) < 1e-12


def test_submax_deviates_and_table():
    m = wilcoxon_sensitivity_moments(10)
    r = submax_test([40, 30], [m.mu] * 2, [m.nu] * 2, submax_comparisons(1), nsim=20000)
    assert abs(r.D[0] - (70 - 2 * m.mu) / math.sqrt(2 * m.nu)) < 1e-12
    assert abs(r.D[1] - (40 - m.mu) / math.sqrt(m.nu)) < 1e-12
    assert abs(r.rho[0][1] - math.sqrt(0.5)) < 1e-12 and abs(r.rho[1][2]) < 1e-15
    assert abs(r.kappa - 2.03) < 0.03  # Table 11.5, p = 1, K = 3
    r2 = submax_test([1] * 4, [0] * 4, [1] * 4, submax_comparisons(2), nsim=30000)
    assert abs(r2.kappa - 2.20) < 0.03  # Table 11.5, p = 2, K = 5
