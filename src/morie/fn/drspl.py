# morie.fn -- slice s03 (rootcoder007/morie)
"""Split-sample (cross-fitted) doubly robust DiD.

Source consulted (FETCHED): Sant'Anna, P. H. C. and Zhao, J. (2020),
*Journal of Econometrics* 219(1), 101-122 (arXiv:1812.01723), equations
(2.6)-(2.7) for the estimand; and Chernozhukov, V. et al. (2018).
Double/debiased machine learning for treatment and structural
parameters.  *The Econometrics Journal* 21(1), C1-C68
(arXiv:1608.00060), whose definition 3.1 is the cross-fitting device:
the nuisance functions pi and mu_0 are estimated on the complement of
each fold and evaluated on the fold itself, so the estimator stays
Neyman-orthogonal without a Donsker condition on the nuisance class.

DETERMINISM.  Folds are assigned by index modulo K -- observation i goes
to fold i mod K.  No permutation, no seed: the split is a function of
the data ordering alone, and both arms produce the same folds.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["dr_did_split_sample"]


def dr_did_split_sample(y, D, X=None, K=5, y0=None):
    """Cross-fitted DR-DiD over K deterministic folds.

    Returns
    -------
    RichResult with payload:
        estimate  : the cross-fitted ATT
        se        : influence-function standard error
        fold_tau  : per-fold estimate
        fold_n    : per-fold sample size
        full_tau  : the un-split estimate, for comparison
    """
    dy = k.vec(y)
    if y0 is not None:
        y00 = k.vec(y0)
        dy = [dy[i] - y00[i] for i in range(len(dy))]
    d = k.vec(D)
    n = len(dy)
    KK = int(K)
    Z = k.design(X, n)
    fold = [i % KK for i in range(n)]
    inf = [0.0] * n
    ftau = []
    fn = []
    num = 0.0
    for f in range(KK):
        tr_i = [i for i in range(n) if fold[i] != f]
        te_i = [i for i in range(n) if fold[i] == f]
        if not te_i or not tr_i:
            ftau.append(float("nan"))
            fn.append(len(te_i))
            continue
        gam = k.logit_irls([Z[i] for i in tr_i], [d[i] for i in tr_i], 60)
        Z0 = [Z[i] for i in tr_i if d[i] < 0.5]
        y0v = [dy[i] for i in tr_i if d[i] < 0.5]
        b0 = k.lstsq(Z0, y0v) if Z0 else [0.0] * len(Z[0])
        s1 = 0.0
        s0 = 0.0
        pis = {}
        mus = {}
        for i in te_i:
            e = 0.0
            m = 0.0
            for j in range(len(gam)):
                e += Z[i][j] * gam[j]
                m += Z[i][j] * b0[j]
            p = k.sigmoid(e)
            pis[i] = p
            mus[i] = m
            s1 += d[i]
            s0 += p * (1.0 - d[i]) / (1.0 - p)
        t = 0.0
        for i in te_i:
            w1 = d[i] / s1 if s1 > 0.0 else 0.0
            w0 = (pis[i] * (1.0 - d[i]) / (1.0 - pis[i]) / s0) if s0 > 0.0 else 0.0
            c = (w1 - w0) * (dy[i] - mus[i])
            t += c
            inf[i] = len(te_i) * c
        ftau.append(t)
        fn.append(len(te_i))
        num += len(te_i) * t
    est = num / n if n else float("nan")
    for i in range(n):
        inf[i] -= est
    v = 0.0
    for x in inf:
        v += x * x
    full = k.drdid_panel(dy, d, X)
    return RichResult(
        title="Cross-fitted DR-DiD",
        summary_lines=[("ATT", est), ("folds", KK)],
        payload={
            "estimate": est,
            "se": (v / (n * n)) ** 0.5 if n else float("nan"),
            "fold_tau": ftau,
            "fold_n": fn,
            "full_tau": full["tau"],
            "n": n,
            "K": KK,
            "method": "Cross-fitted DR-DiD (Sant'Anna and Zhao 2020; Chernozhukov et al. 2018 def. 3.1)",
        },
    )


def cheatsheet():
    return "drspl: Split-sample DR-DiD"
