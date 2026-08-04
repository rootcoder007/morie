# morie.fn -- function file (rootcoder007/morie)
"""Phenotype QC: Box-Cox then Tukey fences."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["phenotype_qc"]


def phenotype_qc(y, k=1.5, lambdas=None):
    """Normalise a phenotype, then flag what is still extreme.

    Order matters here.  Flagging outliers first would throw away the
    right tail of a skewed phenotype -- values that are perfectly
    ordinary once the scale is fixed -- so the transform is chosen
    first, on all the data, and the fences are applied to the
    transformed values.  Box-Cox is fitted by profile likelihood over a
    fixed grid rather than an optimiser, which keeps the answer
    reproducible and, since lambda is reported to two decimals in
    practice anyway, loses nothing.

    Formula: ``y(lambda) = (y^lambda - 1) / lambda`` for
    ``lambda != 0`` and ``log y`` at zero, with lambda maximising the
    profile log-likelihood
    ``-n/2 log(sigma_hat^2(lambda)) + (lambda - 1) sum log y_i``;
    fences are then Tukey ``[H_L - k s, H_U + k s]``.

    Parameters
    ----------
    y : array-like
        Strictly positive phenotype values.
    k : float, default 1.5
        Tukey fence multiplier.
    lambdas : array-like, optional
        Grid of lambda values; ``-2`` to ``2`` in steps of 0.05 by
        default.

    Returns
    -------
    RichResult
        ``estimate`` (fitted lambda), ``loglik``, ``n_out``, ``flags``,
        ``lower``, ``upper``, ``transformed``, ``n``.

    References
    ----------
    Tukey, J. W. (1977).  Exploratory Data Analysis.  Addison-Wesley --
    the fences.  The transform is Box, G. E. P. & Cox, D. R. (1964), An
    analysis of transformations, Journal of the Royal Statistical
    Society B 26:211-252, whose profile likelihood is equation (9) of
    that paper.
    """
    v = C.vec(y)
    n = len(v)
    if min(v) <= 0.0:
        raise ValueError("phenotype values must be strictly positive")
    if lambdas is None:
        lambdas = [(-40 + i) * 0.05 for i in range(81)]
    slog = sum(math.log(t) for t in v)
    best_l, best_ll = 0.0, float("-inf")
    for lam in lambdas:
        if lam == 0.0:
            z = [math.log(t) for t in v]
        else:
            z = [(t ** lam - 1.0) / lam for t in v]
        m = sum(z) / n
        s2 = sum((t - m) ** 2 for t in z) / n
        ll = -0.5 * n * math.log(s2) + (lam - 1.0) * slog
        if ll > best_ll:
            best_ll, best_l = ll, lam
    lam = best_l
    z = [math.log(t) for t in v] if lam == 0.0 else [(t ** lam - 1.0) / lam for t in v]
    s = sorted(z)
    n4 = _floor((n + 3) / 2.0) / 2.0
    def at(d):
        return 0.5 * (s[int(_floor(d)) - 1] + s[int(_ceil(d)) - 1])
    hl, hu = at(n4), at(n + 1 - n4)
    spread = hu - hl
    lo, hi = hl - k * spread, hu + k * spread
    flags = [1.0 if (t < lo or t > hi) else 0.0 for t in z]
    return RichResult(payload={
        "estimate": lam, "loglik": best_ll, "n_out": int(sum(flags)),
        "flags": flags, "lower": lo, "upper": hi, "transformed": z, "n": n,
        "method": "Box-Cox transform then Tukey fences"})


def _floor(v):
    return float(int(v) if v >= 0 or v == int(v) else int(v) - 1)


def _ceil(v):
    return float(int(v) if v <= 0 or v == int(v) else int(v) + 1)


phenotypeqc = phenotype_qc


def cheatsheet():
    return "pheno2: Phenotype QC: Box-Cox then Tukey fences."
