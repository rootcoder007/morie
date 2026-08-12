"""Tests for deseq2 (Love, Huber & Anders 2014)."""

import math

from morie.fn.deseq2 import (benjamini_hochberg, cox_reid_loglik, deseq2,
                             deseq2_differential, dispersion_trend,
                             nb_glm_fit, size_factors, trigamma)


def _lcg(seed):
    state = [seed]

    def f():
        state[0] = (1103515245 * state[0] + 12345) % (1 << 31)
        return state[0] / float(1 << 31)
    return f


def _sim(n_genes=120, n_rep=8, seed=2024, up=15, down=15):
    rnd = _lcg(seed)

    def normal():
        u1 = max(rnd(), 1e-12)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * rnd())

    def gamma(shape):
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            z = normal()
            v = (1.0 + c * z) ** 3
            if v <= 0:
                continue
            if math.log(max(rnd(), 1e-12)) < 0.5 * z * z + d - d * v + \
                    d * math.log(v):
                return d * v

    def nb(mu, alpha):
        shape = 1.0 / alpha
        lam = mu * gamma(shape) / shape
        if lam > 400:
            return int(lam + math.sqrt(lam) * normal())
        L = math.exp(-lam)
        k, pp = 0, 1.0
        while True:
            pp *= max(rnd(), 1e-12)
            if pp <= L:
                return k
            k += 1

    counts, truth = [], []
    for g in range(n_genes):
        base = math.exp(2.0 + 4.0 * rnd())
        alpha = math.exp(math.log(3.0 / base + 0.08) + 0.5 * normal())
        lfc = 2.0 if g < up else (-2.0 if g < up + down else 0.0)
        counts.append([nb(base, alpha) for _ in range(n_rep)] +
                      [nb(base * 2 ** lfc, alpha) for _ in range(n_rep)])
        truth.append(lfc != 0.0)
    return counts, truth, ["A"] * n_rep + ["B"] * n_rep


def test_trigamma_closed_forms():
    assert abs(trigamma(1.0) - math.pi ** 2 / 6.0) < 1e-12
    assert abs(trigamma(0.5) - math.pi ** 2 / 2.0) < 1e-12
    assert abs(trigamma(2.0) - (math.pi ** 2 / 6.0 - 1.0)) < 1e-12
    for x in (0.3, 1.7, 4.2):
        assert abs(trigamma(x + 1.0) - (trigamma(x) - 1.0 / x ** 2)) < 1e-12


def test_size_factors_on_proportional_samples():
    profile = [10.0, 55.0, 3.0, 900.0, 120.0]
    mult = [0.5, 1.0, 2.0, 4.0]
    sf = size_factors([[v * c for c in mult] for v in profile])
    gm = math.exp(sum(math.log(c) for c in mult) / len(mult))
    for j, c in enumerate(mult):
        assert abs(sf[j] - c / gm) < 1e-12


def test_nb_glm_satisfies_the_score_equation():
    K = [12.0, 30.0, 21.0, 140.0, 90.0, 200.0]
    X = [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 3
    s = [0.8, 1.0, 1.25, 0.9, 1.1, 1.0]
    for alpha in (1e-6, 0.4, 3.0):
        fit = nb_glm_fit(K, X, alpha, s)
        for r in range(2):
            score = sum(X[j][r] * (K[j] - fit["mu"][j]) /
                        (1.0 + alpha * fit["mu"][j]) for j in range(6))
            assert abs(score) < 1e-6
        # equal size factors collapse it to the group means
        eq = nb_glm_fit(K, X, alpha, [1.0] * 6)
        assert abs(math.exp(eq["beta"][0]) - sum(K[:3]) / 3.0) < 1e-6


def test_cox_reid_matches_the_formula():
    K = [12.0, 30.0, 21.0, 140.0, 90.0, 200.0]
    X = [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 3
    mu = nb_glm_fit(K, X, 0.1, [1.0] * 6)["mu"]
    alpha = 0.15
    r = 1.0 / alpha
    ll = sum(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1.0) +
             r * math.log(r / (r + m)) + k * math.log(m / (r + m))
             for k, m in zip(K, mu))
    W = [1.0 / (1.0 / m + alpha) for m in mu]
    a = sum(W[j] * X[j][0] ** 2 for j in range(6))
    b = sum(W[j] * X[j][0] * X[j][1] for j in range(6))
    d = sum(W[j] * X[j][1] ** 2 for j in range(6))
    want = ll - 0.5 * math.log(a * d - b * b)
    assert abs(cox_reid_loglik(alpha, K, mu, X) - want) < 1e-9


def test_trend_recovers_its_own_curve():
    mus = [10.0 * 1.35 ** t for t in range(40)]
    tr = dispersion_trend(mus, [3.0 / mu + 0.02 for mu in mus])
    assert abs(tr["a1"] - 3.0) < 1e-6
    assert abs(tr["a0"] - 0.02) < 1e-8


def test_benjamini_hochberg():
    p = [0.001, 0.02, 0.03, 0.5, 0.9]
    adj = benjamini_hochberg(p)
    assert all(adj[i] >= p[i] - 1e-12 for i in range(5))
    assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(4))
    assert abs(adj[0] - 0.005) < 1e-12


def test_dispersion_shrinkage_rules():
    counts, _, design = _sim()
    res = deseq2(counts, design)
    n = len(counts)
    assert abs(res["sigma_d2"] -
               max(res["s_lr"] ** 2 - trigamma((16 - 2) / 2.0), 0.25)) < 1e-12
    for i in range(n):
        gw = res["dispersion_gene_wise"][i]
        fit = res["dispersion_fit"][i]
        if res["dispersion_outlier"][i]:
            assert abs(res["dispersion"][i] - gw) < 1e-12
            assert math.log(gw) > math.log(fit) + 2.0 * res["s_lr"]
        else:
            assert (min(gw, fit) - 1e-9 <= res["dispersion"][i] <=
                    max(gw, fit) + 1e-9)


def test_lfc_shrinkage_and_testing():
    counts, truth, design = _sim()
    res = deseq2(counts, design)
    n = len(counts)
    assert all(abs(res["log_fold_change"][i]) <= abs(res["lfc_mle"][i]) + 1e-9
               for i in range(n))
    # quantile matching sets the prior width
    vals = sorted(abs(v) * math.log(2.0) for v in res["lfc_mle"])
    pos = 0.95 * (len(vals) - 1)
    lo = int(math.floor(pos))
    emp = vals[lo] + (pos - lo) * (vals[min(lo + 1, len(vals) - 1)] - vals[lo])
    assert abs(res["prior_sigma"][1] * 1.959963984540054 - emp) < 1e-6
    # Wald test and BH
    for i in range(n):
        if res["lfc_se"][i] > 0:
            assert abs(res["stat"][i] * res["lfc_se"][i] -
                       res["log_fold_change"][i]) < 1e-9
        assert res["padj"][i] >= res["pvalue"][i] - 1e-12
    called = [i for i in range(n) if res["padj"][i] < 0.1]
    tp = sum(1 for i in called if truth[i])
    assert tp >= 20
    assert (len(called) - tp) / float(max(len(called), 1)) < 0.2


def test_beta_prior_off_gives_the_mle():
    counts, _, design = _sim(n_genes=60)
    res = deseq2(counts, design, beta_prior=False)
    assert all(abs(res["log_fold_change"][i] - res["lfc_mle"][i]) < 1e-12
               for i in range(60))


def test_validation():
    counts, _, design = _sim(n_genes=20)
    for call in (lambda: deseq2([], ["A"]),
                 lambda: deseq2(counts, ["A"] * 16),
                 lambda: deseq2(counts, ["A"] * 4 + ["B"] * 4),
                 lambda: size_factors([[-1.0, 2.0], [3.0, 4.0]]),
                 lambda: deseq2(counts, design, contrast=[1.0, 0.0, 0.0]),
                 lambda: deseq2(counts, design, size=[0.0] * 16)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert deseq2_differential is deseq2
