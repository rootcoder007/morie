"""Shared generator for the causal-forest family tests.

Not a test module -- imported by test_plrgrf, test_crfsel, test_itrgrf,
test_ipwgrf and test_frfgrf. The design puts a STRONG confounding
surface next to a weak treatment effect, which is the case that
separates a locally centered forest from an uncentered one.

`m_scale` defaults to 1 so the nuisance forests can actually learn the
confounding surface at these sample sizes; plrgrf asks for m_scale=3
because strong confounding is the case local centering exists for.
`randomized=True` sets e = 0.5, which is where the augmented IPW score
is identified without relying on the forest to recover a propensity.
"""
import math
from morie.fn import _array_core as np


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


def confounded(n, seed, tau_scale=1.0, m_scale=1.0,
               randomized=False):
    rng = np.random.default_rng(seed)
    X = [[rng.standard_normal() for _ in range(3)] for _ in range(n)]
    e = ([0.5] * n if randomized
         else [expit(1.2 * X[i][0]) for i in range(n)])
    W = [1.0 if float(rng.uniform()) < e[i] else 0.0 for i in range(n)]
    m = [m_scale * X[i][0] + 1.0 * X[i][1] for i in range(n)]
    tau = [tau_scale * (0.5 + X[i][1]) for i in range(n)]
    y = [m[i] + tau[i] * (W[i] - e[i]) + 0.3 * rng.standard_normal()
         for i in range(n)]
    return {"X": X, "y": y, "W": W, "e": e, "tau": tau, "m": m, "n": n}


def clustered(m, per, seed, icc=2.0):
    rng = np.random.default_rng(seed)
    X, y, lab = [], [], []
    for c in range(m):
        u = icc * rng.standard_normal()
        for _ in range(per):
            x = [rng.standard_normal() for _ in range(2)]
            X.append(x)
            y.append(1.0 * x[0] + u + 0.3 * rng.standard_normal())
            lab.append("c%02d" % c)
    return {"X": X, "y": y, "cluster": lab, "n": len(y), "m": m}
