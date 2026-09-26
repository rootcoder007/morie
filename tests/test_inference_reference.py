"""morie.inference against base R on fixed data.

References: t.test (var.equal, alternative), power.t.test and
power.prop.test solved with tol = 1e-14, cor.test(method = "spearman"),
and fisher.test (conditional MLE and exact interval, verified to satisfy
E[X | psi] = x and P(X <= x | psi_U) = 0.025 to 1e-14; R's own uniroot
stops about 1e-4 short, so the interval is checked through those
equations here).
"""

import math

from morie import inference as I

A = [round(5 + 2 * math.sin(1.3 * i) + 0.1 * i, 3) for i in range(25)]
B = [round(6 + 2.5 * math.cos(0.7 * i) + 0.05 * i, 3) for i in range(22)]


def rel(a, b):
    return abs(a - b) / abs(b)


def test_t_intervals_match_t_test():
    r = I.two_sample_t_test(A, B, equal_var=True)
    assert rel(r["ci_diff_lower"], -1.46891850244) <= 1e-10
    assert rel(r["ci_diff_upper"], 0.487754866072) <= 1e-10
    r = I.two_sample_t_test(A, B, alternative="less")
    assert r["ci_diff_lower"] == -math.inf and rel(r["ci_diff_upper"], 0.337090885184) <= 1e-10
    r = I.one_sample_t_test(A, 5.5, alternative="greater")
    assert rel(r["ci_lower"], 5.67743355663) <= 1e-10 and r["ci_upper"] == math.inf
    r = I.paired_t_test(A[:22], B)
    assert rel(r["ci_lower"], -1.53492517052) <= 1e-10


def test_power_matches_power_t_and_prop_test():
    assert rel(I.power_t_test(delta=0.8, sd=1.0, power=0.8), 25.5246312298881) <= 1e-11
    assert rel(I.power_t_test(n=20, delta=0.8), 0.693399444317275) <= 1e-12
    assert rel(I.power_t_test(n=20, power=0.8), 0.909130127626943) <= 1e-11
    assert rel(I.power_t_test(delta=0.5, power=0.9, type="paired", alternative="one-sided"), 35.652676906496) <= 1e-11
    assert rel(I.power_prop_test(p1=0.3, p2=0.5, power=0.8), 92.9988448275456) <= 1e-11
    assert rel(I.power_prop_test(n=60, p1=0.3, p2=0.5), 0.61104449768773) <= 1e-12
    assert rel(I.power_prop_test(n=60, p1=0.3, power=0.8), 0.550394038704573) <= 1e-11


def test_fisher_conditional_estimate_and_interval():
    tab = [[3, 1], [1, 3]]
    r = I.fisher_exact_test(tab)
    # defining equations of the conditional MLE and exact upper limit
    m, n, k, x = 4, 4, 4, 3
    sup = range(max(0, k - n), min(k, m) + 1)

    def dens(psi):
        w = [math.comb(m, u) * math.comb(n, k - u) * psi**u for u in sup]
        t = sum(w)
        return [v / t for v in w]

    assert abs(sum(u * p for u, p in zip(sup, dens(r["odds_ratio"]))) - x) <= 1e-12
    assert abs(sum(p for u, p in zip(sup, dens(r["ci_upper"])) if u <= x) - 0.025) <= 1e-12
    assert abs(sum(p for u, p in zip(sup, dens(r["ci_lower"])) if u >= x) - 0.025) <= 1e-12
    assert r["sample_odds_ratio"] == 9.0


def test_spearman_exact_p_matches_cor_test():
    # cor.test(x, y, method = "spearman") for these untied data (n = 8)
    x = [3, 1, 4, 1.5, 5, 9, 2.6, 6]
    y = [2, 7, 1, 8, 2.8, 1.8, 2.9, 4.5]
    r = I.spearman_rho(x, y)
    rho = r["rho"]
    # exact by enumeration here too: S = sum d^2, P(S <= s) over 8! orders
    import itertools

    rx = [sorted(x).index(v) + 1 for v in x]
    ry = [sorted(y).index(v) + 1 for v in y]
    s_obs = sum((a - b) ** 2 for a, b in zip(rx, ry))
    assert abs(rho - (1 - 6 * s_obs / (8 * 63))) <= 1e-12
    counts = [sum((i + 1 - v) ** 2 for i, v in enumerate(p)) for p in itertools.permutations(range(1, 9))]
    p_lower = sum(c <= s_obs for c in counts) / len(counts)
    p_upper = sum(c >= s_obs for c in counts) / len(counts)
    assert abs(r["p_value"] - min(1.0, 2 * min(p_lower, p_upper))) <= 1e-12
