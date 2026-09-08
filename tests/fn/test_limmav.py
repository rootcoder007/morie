"""Tests for limmav (Law, Chen, Shi & Smyth 2014, voom)."""

import math

from morie.fn.limmav import (digamma, ebayes, limma_voom, limmav, log_cpm,
                             lowess, trigamma, trigamma_inverse, voom,
                             voom_weights)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _nb(rnd, mu, disp):
    shape = 1.0 / disp
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        u1 = max(rnd(), 1e-12)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * rnd())
        v = (1.0 + c * z) ** 3
        if v <= 0:
            continue
        if math.log(max(rnd(), 1e-12)) < 0.5 * z * z + d - d * v + \
                d * math.log(v):
            lam = mu * d * v / shape
            break
    if lam > 400:
        u1 = max(rnd(), 1e-12)
        return max(0, int(lam + math.sqrt(lam) *
                          math.sqrt(-2 * math.log(u1)) *
                          math.cos(2 * math.pi * rnd())))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= max(rnd(), 1e-12)
        if p <= L:
            return k
        k += 1


LIBS = [8e5, 1.2e6, 1.0e6, 3.0e6, 2.5e6, 2.8e6]
DESIGN = [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 3


def _panel(n_genes=120, seed=11, up=0, down=0):
    rnd = _lcg(seed)
    K, truth = [], []
    for g in range(n_genes):
        base = math.exp(2.0 + 5.0 * rnd())
        disp = 0.05 + 0.15 * rnd()
        lfc = 2.0 if g < up else (-2.0 if g < up + down else 0.0)
        row = [_nb(rnd, base * LIBS[i] / 1e6, disp) for i in range(3)]
        row += [_nb(rnd, base * 2 ** lfc * LIBS[i] / 1e6, disp)
                for i in range(3, 6)]
        K.append(row)
        truth.append(lfc != 0.0)
    return K, truth


def test_log_cpm_formula():
    counts = [[10, 0, 500], [3, 7, 1000]]
    libs = [1000.0, 2000.0, 40000.0]
    y, R = log_cpm(counts, libs)
    for g in range(2):
        for i in range(3):
            want = math.log((counts[g][i] + 0.5) / (libs[i] + 1.0) * 1e6, 2)
            assert abs(y[g][i] - want) < 1e-12
    assert log_cpm(counts)[1] == [13.0, 7.0, 1500.0]


def test_lowess_is_exact_on_a_line_and_robust():
    xs = [t / 10.0 for t in range(30)]
    ys = [3.0 + 2.0 * v for v in xs]
    assert max(abs(a - b) for a, b in zip(lowess(xs, ys), ys)) < 1e-9
    ys2 = list(ys)
    ys2[15] += 50.0
    fit = lowess(xs, ys2, iterations=3)
    assert abs(fit[15] - ys[15]) < 1.0


def test_weights_are_lo_of_the_fitted_log_count_to_the_minus_four():
    K, _ = _panel()
    v = voom_weights(K, DESIGN)
    kx, ky = v["trend_x"], v["trend_y"]

    def lo(t):
        if t <= kx[0]:
            return ky[0]
        if t >= kx[-1]:
            return ky[-1]
        for i in range(len(kx) - 1):
            if kx[i] <= t <= kx[i + 1]:
                if kx[i + 1] - kx[i] < 1e-15:
                    return ky[i]
                f = (t - kx[i]) / (kx[i + 1] - kx[i])
                return ky[i] + f * (ky[i + 1] - ky[i])
        return ky[-1]

    y, R = v["log_cpm"], v["lib_sizes"]
    for g in range(0, len(K), 17):
        ya, yb = sum(y[g][:3]) / 3.0, sum(y[g][3:]) / 3.0
        fitted = [ya] * 3 + [yb] * 3
        for i in range(6):
            lam = fitted[i] + math.log(R[i] + 1.0, 2) - math.log(1e6, 2)
            assert abs(v["weights"][g][i] - 1.0 / lo(lam) ** 4) < 1e-9


def test_mean_log_count_uses_the_geometric_mean_of_lib_plus_one():
    K, _ = _panel()
    v = voom_weights(K, DESIGN)
    y, R = v["log_cpm"], v["lib_sizes"]
    logR = sum(math.log(t + 1.0, 2) for t in R) / 6
    for g in (0, 37, 99):
        want = sum(y[g]) / 6 + logR - math.log(1e6, 2)
        assert abs(v["mean_log_count"][g] - want) < 1e-12


def test_the_trend_is_steep_where_poisson_noise_dominates():
    rnd = _lcg(3)

    def pois(lam):
        if lam > 400:
            u1 = max(rnd(), 1e-12)
            return max(0, int(lam + math.sqrt(lam) *
                              math.sqrt(-2 * math.log(u1)) *
                              math.cos(2 * math.pi * rnd())))
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= max(rnd(), 1e-12)
            if p <= L:
                return k
            k += 1

    K = []
    for _ in range(200):
        base = math.exp(0.5 + 7.0 * rnd())
        K.append([pois(base * LIBS[i] / 1e6) for i in range(6)])
    v = voom_weights(K, DESIGN)
    assert v["trend_y"][0] > 3.0 * v["trend_y"][-1]
    order = sorted(range(200), key=lambda g: v["mean_log_count"][g])
    low = sum(sum(v["weights"][g]) for g in order[:50]) / 50
    high = sum(sum(v["weights"][g]) for g in order[-50:]) / 50
    assert high > 50.0 * low


def test_weighted_and_unweighted_fits_differ():
    K, _ = _panel()
    w = limmav(K, DESIGN)
    u = limmav(K, DESIGN, weights=False)
    assert max(abs(w["estimate"][g] - u["estimate"][g])
               for g in range(len(K))) > 1e-6


def test_t_distribution_and_testing():
    K, _ = _panel()
    r = limmav(K, DESIGN)
    assert r["df"] == 4
    for g in (0, 50, 99):
        assert abs(r["t"][g] * r["se"][g] - r["estimate"][g]) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in r["pvalue"])
    assert all(r["padj"][g] >= r["pvalue"][g] - 1e-12
               for g in range(len(K)))


def test_detects_planted_genes():
    K, truth = _panel(n_genes=200, seed=77, up=20, down=20)
    r = limmav(K, DESIGN)
    called = [g for g in range(200) if r["padj"][g] < 0.1]
    tp = sum(1 for g in called if truth[g])
    assert tp >= 15
    assert (len(called) - tp) / float(max(len(called), 1)) < 0.25
    assert all(r["estimate"][g] > 0 for g in range(20))
    assert all(r["estimate"][g] < 0 for g in range(20, 40))


def test_validation():
    K, _ = _panel(n_genes=20)
    for call in (lambda: limmav([], ["A", "B"]),
                 lambda: log_cpm([[-1, 2]], [10.0, 10.0]),
                 lambda: log_cpm([[1, 2]], [0.0, 10.0]),
                 lambda: limmav(K, [[1.0, 0.0]] * 4),
                 lambda: limmav(K, ["A"] * 6),
                 lambda: limmav(K, DESIGN, contrast=[1.0]),
                 lambda: lowess([1.0, 2.0], [1.0, 2.0], span=0.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert voom is limmav and limma_voom is limmav


def test_digamma_and_trigamma_against_closed_forms():
    assert abs(digamma(1.0) + 0.5772156649015329) < 1e-10
    assert abs(digamma(0.5) -
               (-0.5772156649015329 - 2.0 * math.log(2.0))) < 1e-10
    for x in (0.3, 1.7, 4.2, 30.0):
        assert abs(digamma(x + 1.0) - digamma(x) - 1.0 / x) < 1e-10
    assert abs(trigamma(1.0) - math.pi ** 2 / 6.0) < 1e-12


def test_trigamma_inverse_round_trip():
    for y in (0.4, 1.0, 3.7, 25.0):
        assert abs(trigamma_inverse(trigamma(y)) - y) < 1e-6
    # The paper's overflow guards.
    assert abs(trigamma_inverse(1e8) - 1e-4) < 1e-6
    assert abs(trigamma_inverse(1e-8) - 1e8) < 1.0


def _chisq(rnd, df):
    tot = 0.0
    for _ in range(int(df)):
        u1 = max(rnd(), 1e-12)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * rnd())
        tot += z * z
    return tot


def test_ebayes_recovers_its_own_hyperparameters():
    d0, s0, dg = 6.0, 2.0, 8
    rnd = _lcg(5)
    s2 = []
    for _ in range(3000):
        sigma2 = d0 * s0 / _chisq(rnd, d0)
        s2.append(sigma2 * _chisq(rnd, dg) / dg)
    eb = ebayes(s2, dg)
    assert abs(eb["d0"] - d0) < 0.6
    assert abs(eb["s0_sq"] - s0) < 0.2
    for g in (0, 100, 2999):
        want = (eb["d0"] * eb["s0_sq"] + dg * s2[g]) / (eb["d0"] + dg)
        assert abs(eb["s2_post"][g] - want) < 1e-12
        assert abs(eb["df_total"][g] - (dg + eb["d0"])) < 1e-12
    # Every posterior variance lies between the prior and the gene's own.
    assert all(min(eb["s0_sq"], s2[g]) - 1e-12 <= eb["s2_post"][g] <=
               max(eb["s0_sq"], s2[g]) + 1e-12 for g in range(len(s2)))


def test_ebayes_degenerate_branch():
    flat = ebayes([1.3] * 200, 8)
    assert flat["d0"] == float("inf")
    assert flat["no_gene_variation"]
    assert all(abs(v - flat["s2_post"][0]) < 1e-12 for v in flat["s2_post"])


def test_moderation_is_on_by_default_and_can_be_turned_off():
    K, _ = _panel(n_genes=300, seed=31)
    mod = limmav(K, DESIGN)
    ord_ = limmav(K, DESIGN, moderate=False)
    assert mod["moderated"] and not ord_["moderated"]
    assert mod["d0"] > 0 and mod["s0_sq"] > 0
    assert ord_["d0"] is None and ord_["df_total"] is None
    assert mod["df_total"][0] > ord_["df"]
    # Moderation touches the denominator only.
    assert all(abs(mod["estimate"][g] - ord_["estimate"][g]) < 1e-12
               for g in range(len(K)))
    tiny = sorted(range(len(K)), key=lambda g: mod["s2_gene"][g])[:30]
    assert all(mod["s2_post"][g] > mod["s2_gene"][g] for g in tiny)
    assert sum(1 for g in tiny
               if abs(mod["t"][g]) < abs(ord_["t"][g])) >= 28
    big = sorted(range(len(K)), key=lambda g: -mod["s2_gene"][g])[:30]
    assert all(mod["s2_post"][g] < mod["s2_gene"][g] for g in big)


def test_moderation_buys_power_at_n_equals_three():
    K, truth = _panel(n_genes=200, seed=77, up=20, down=20)
    mod = limmav(K, DESIGN)
    ord_ = limmav(K, DESIGN, moderate=False)
    tp_m = sum(1 for g in range(200)
               if mod["padj"][g] < 0.1 and truth[g])
    tp_o = sum(1 for g in range(200)
               if ord_["padj"][g] < 0.1 and truth[g])
    assert tp_m >= tp_o + 10
