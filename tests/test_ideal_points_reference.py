"""Binary probit EM ideal points (emIRT::binIRT with asEM = TRUE)."""

import math

from morie._spatial_voting import em_irt
from morie.fn._rng import random_normal, random_uniform


def votes():
    U = [float(v) for v in random_uniform(4000, seed=11, stream=0)]
    Z = [float(v) for v in random_normal(4000, seed=11, stream=1)]
    V = []
    for i in range(30):
        row = []
        for j in range(20):
            m = 0.3 * Z[100 + j] + (1.0 + Z[200 + j]) * Z[i] + 0.5 * Z[300 + j] * Z[400 + i]
            p = 0.5 * math.erfc(-m / math.sqrt(2))
            row.append(float("nan") if U[2000 + i * 20 + j] < 0.05 else (1.0 if U[i * 20 + j] < p else 0.0))
        V.append(row)
    return V


def log_posterior(V, a, b, x):
    lp = 0.0
    for i, row in enumerate(V):
        for j, y in enumerate(row):
            if y == y:
                m = a[j] + sum(bd * xd for bd, xd in zip(b[j], x[i]))
                lp += math.log(0.5 * math.erfc(-(m if y == 1 else -m) / math.sqrt(2)))
    lp -= 0.5 * sum(v * v for r in x for v in r)
    lp -= 0.5 * (sum(v * v for v in a) + sum(v * v for r in b for v in r)) / 25
    return lp


def test_matches_binIRT():
    V = votes()
    r = em_irt(V)
    # emIRT::binIRT(.starts = same x, alpha = beta = 0; makePriors; default control): 83 iterations
    assert r["iterations"] == 83 and r["converged"]
    t = em_irt(V, max_iter=20000, tol=1e-13, conv="abs")
    # binIRT(convtype = 2, thresh = 1e-13): x[1] -0.5391190086, alpha[1:2], beta[1]
    assert abs(t["ideal_points"][0][0] + 0.5391190086) < 1e-9
    assert abs(t["difficulty"][0] - 0.6150891743) < 1e-9 and abs(t["difficulty"][1] + 0.4241460078) < 1e-9
    assert abs(t["discrimination"][0][0] + 6.5721230364) < 1e-9
    assert abs(t["log_lik"] - (-267.3624873117)) < 1e-9


def test_two_dimensional_fixed_point_is_posterior_mode():
    V = votes()
    r = em_irt(V, n_dims=2, max_iter=20000, tol=1e-12, conv="abs")
    a, b, x = r["difficulty"], r["discrimination"], r["ideal_points"]
    h = 1e-5
    for which, idx in (("x", (0, 1)), ("x", (7, 0)), ("b", (3, 1)), ("a", (5,))):

        def lp(delta):
            aa, bb, xx = list(a), [list(v) for v in b], [list(v) for v in x]
            if which == "a":
                aa[idx[0]] += delta
            else:
                (xx if which == "x" else bb)[idx[0]][idx[1]] += delta
            return log_posterior(V, aa, bb, xx)

        assert abs((lp(h) - lp(-h)) / (2 * h)) < 1e-5
    base = log_posterior(V, a, b, x)
    assert base > log_posterior(V, a, b, [[v + 0.01 for v in row] for row in x])


def test_cjr_gibbs_chain_matches_r_arm():
    from morie._spatial_voting import cjr_irt

    V = votes()
    f = cjr_irt(V, n_samples=200, burn_in=50, seed=5)
    # morie_spatial_voting_cjr_irt(V, n_samples = 200, burn_in = 50, seed = 5): same Philox draws
    got = [f["ideal_point_mean"][i][0] for i in range(3)] + f["alpha_mean"][:2]
    got += [f["beta_mean"][0][0], f["ideal_point_sd"][0][0]]
    ref = [
        -0.577500384336,
        -0.678132861519,
        -0.176501975313,
        -1.12013879797,
        0.540891603986,
        -9.235856442311,
        0.297621796656,
    ]
    assert all(abs(g - r) < 1e-9 for g, r in zip(got, ref))
    assert len(f["ideal_point_chain"]) == 200 and len(f["beta_chain"][0]) == 20


def test_cjr_far_tail_helpers_match_r():
    from morie._spatial_voting import _cjr_log_phi_far, _cjr_log_tail_quantile

    # pnorm(-50, log.p = TRUE); qnorm(log(0.3) + pnorm(-50, log.p = TRUE), log.p = TRUE)
    assert abs(_cjr_log_phi_far(-50.0) - (-1254.831361139419869)) < 1e-9
    assert abs(_cjr_log_tail_quantile(math.log(0.3) + _cjr_log_phi_far(-50.0)) - (-50.024064049676952)) < 1e-9
    assert abs(_cjr_log_phi_far(-30.5) - (-469.462737322912119)) < 1e-9
