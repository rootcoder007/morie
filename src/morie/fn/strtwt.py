# morie.fn -- function file (rootcoder007/morie)
"""Stabilized inverse-probability-of-treatment weights within strata."""

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["stratified_weights"]


def stratified_weights(A, H=None, S=None):
    """Stabilized treatment weights whose numerator conditions on strata.

    The unstabilized weight ``1/f(A|H,S)`` has no upper bound: a unit
    whose observed treatment was nearly impossible given its history
    carries nearly infinite weight.  Cole and Hernan's stabilization
    divides it by the probability of the same treatment under a model
    that drops the history, which leaves the weighted pseudo-population
    unchanged but shrinks the weights towards one.  Keeping the stratum
    variables in the numerator makes the weights valid for a marginal
    structural model that itself conditions on the stratum.

    Both probability models are logistic regressions fitted by IRLS, so
    the whole procedure is deterministic and the two language arms land
    on identical numbers.

    Formula: ``sw_i = f(A_i | S_i) / f(A_i | H_i, S_i)``, with ``f`` the
    Bernoulli density evaluated at the observed treatment.

    Parameters
    ----------
    A : array-like
        Binary treatment, coded 0/1.
    H : array-like, optional
        History covariates, ``n x p``.  If omitted the two models
        coincide and every weight is exactly one.
    S : array-like, optional
        Stratum covariates, ``n x q``, entering both models.

    Returns
    -------
    RichResult
        ``estimate`` (mean stabilized weight), ``weights``,
        ``unstabilized``, ``num`` (numerator probabilities), ``den``
        (denominator probabilities), ``max``, ``min``, ``sd``, ``n``,
        ``method``.

    References
    ----------
    Cole, S. R. & Hernan, M. A. (2008).  Constructing inverse probability
    weights for marginal structural models.  American Journal of
    Epidemiology 168(6):656-664.  <https://doi.org/10.1093/aje/kwn164>
    """
    a = C.vec(A)
    n = len(a)
    if n == 0:
        raise ValueError("stratified_weights: A is empty")
    for v in a:
        if v not in (0.0, 1.0):
            raise ValueError("stratified_weights: A must be coded 0/1")
    Sm = _cols(S, n, "S")
    Hm = _cols(H, n, "H")
    Zn = [[1.0] + Sm[i] for i in range(n)]
    Zd = [[1.0] + Sm[i] + Hm[i] for i in range(n)]
    pn = _fit(Zn, a)
    pd = _fit(Zd, a)
    num = [pn[i] if a[i] > 0.5 else 1.0 - pn[i] for i in range(n)]
    den = [pd[i] if a[i] > 0.5 else 1.0 - pd[i] for i in range(n)]
    sw = [num[i] / den[i] for i in range(n)]
    uw = [1.0 / den[i] for i in range(n)]
    m = sum(sw) / n
    v = sum((x - m) ** 2 for x in sw) / (n - 1) if n > 1 else 0.0
    return RichResult(payload={
        "estimate": float(m), "weights": sw, "unstabilized": uw,
        "num": num, "den": den, "max": float(max(sw)), "min": float(min(sw)),
        "sd": float(v ** 0.5), "n": n,
        "method": "sw = f(A|S) / f(A|H,S), stabilized IPTW [Cole & Hernan 2008]"})


def _cols(X, n, nm):
    if X is None:
        return [[] for _ in range(n)]
    M = core.mat(X)
    if len(M) != n:
        raise ValueError("stratified_weights: %s has the wrong number of rows" % nm)
    return [list(r) for r in M]


def _fit(Z, a):
    b = core.logit_irls(Z, a, 60)
    return [min(max(core.sigmoid(sum(Z[i][j] * b[j] for j in range(len(b)))), 1e-12), 1.0 - 1e-12)
            for i in range(len(Z))]


# CANONICAL TEST
# >>> A = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
# >>> S = [[0.0], [0.0], [1.0], [1.0], [0.0], [1.0], [1.0], [0.0]]
# >>> r = stratified_weights(A, None, S)
# >>> assert all(abs(w - 1.0) < 1e-9 for w in r["weights"])   # no history: sw == 1
# >>> assert abs(r["estimate"] - 1.0) < 1e-9


def cheatsheet():
    return "strtwt(A, H, S): stabilized IPTW, f(A|S)/f(A|H,S)."
