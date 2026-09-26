"""Bayesian kriging (eqs 6.94-6.98) and Handcock-Stein against gstat, the printed (6.97) and geoR."""

import math

from morie.fn.bkrnig import bayesian_kriging_nig
from morie.fn.hsbkrg import bayesian_kriging_matern

N = 20
CO = [(10 * ((i * 0.618034) % 1), 10 * ((i * 0.414214 + 0.3) % 1)) for i in range(N)]
Z = [1 + 0.2 * x + math.sin(0.8 * x) + math.cos(0.6 * y) + 0.2 * math.sin(5.3 * i) for i, (x, y) in enumerate(CO)]
X = [[1.0, x] for x, _ in CO]
C0 = [(2.5, 3.5), (7.0, 1.0), (5.0, 8.5)]
X0 = [[1.0, x] for x, _ in C0]


def corr(A, B):
    return [[math.exp(-3 * math.dist(a, b) / 4) for b in B] for a in A]


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_flat_prior_is_gstat_universal_kriging():
    f = bayesian_kriging_nig(X, Z, X0, corr(CO, CO), corr(C0, CO), corr(C0, C0))
    # gstat::krige(z ~ x, model = vgm(1, "Exp", 4 / 3)): var1.pred, var1.var
    pred = [1.7061445705561447, 2.3493210010069245, 1.9095305189366867]
    var = [0.72867075615633337, 0.44013082829933997, 0.76854464567679392]
    assert all(close(a, b, 1e-12) for a, b in zip(f["mean"], pred))
    k = f["a_star"] / (f["df"] - 2)
    assert all(close(float(f["variance"][i][i]), var[i] * k, 1e-12) for i in range(3))


def test_nig_mean_is_the_printed_6_97():
    f = bayesian_kriging_nig(
        X, Z, X0, corr(CO, CO), corr(C0, CO), corr(C0, C0), a=2.0, d=4.0, m=[0.5, 0.1], Qinv=[[2.0, 0.0], [0.0, 5.0]]
    )
    ref = [1.6861663842349945, 2.3517380801871717, 1.8853077971494199]
    assert all(close(a, b, 1e-12) for a, b in zip(f["mean"], ref))
    assert close(f["a_star"], 13.596024854143266, 1e-12) and f["df"] == 24


def test_handcock_stein_matches_geor_krige_bayes():
    h = bayesian_kriging_matern(CO, Z, X, C0, X0, [1 / p for p in (0.5, 1.0, 1.5, 2.0, 2.5)], [1.5])
    # geoR::krige.bayes, matern kappa = 1.5, flat beta, reciprocal sigmasq, phi.discrete uniform
    assert all(
        close(a, b, 1e-9) for a, b in zip(h["mean"], [1.8901783255847411, 2.3797672496277436, 1.6793035375250742])
    )
    assert all(
        close(a, b, 1e-9)
        for a, b in zip(h["variance"], [0.16557142216248577, 0.039438978803821634, 0.19073490148919969])
    )
    post = [r[0] for r in h["posterior"]]
    ref = [0.0056473900854062961, 0.053295324569191373, 0.19400351523484616, 0.33942697726502258, 0.4076267928455336]
    assert all(close(a, b, 1e-9) for a, b in zip(post, ref))
