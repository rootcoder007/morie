# morie.fn -- function file (rootcoder007/morie)
"""Dimension reduction in single-index and multiple-index models.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 1.2 (page 3), Section 2.2 (page 11) and
Section 2.4 (page 18).

A single-index model E(Y|X=x) = G(x'beta) collapses a d-dimensional
covariate onto one index.  The book's two rate statements are:

  * beta is estimable at the parametric rate n^{-1/2} (Section 2.2,
    page 11; Sections 2.5 and 2.6);
  * because beta converges faster than any nonparametric estimator,
    replacing beta by an estimate has no effect on the asymptotics of
    the estimator of G (page 18), so G converges at the ONE
    dimensional nonparametric rate, n^{-s/(2s+1)} -- n^{-2/5} for a
    twice-differentiable G.

A multiple-index model with M indices reduces to an M-dimensional
nonparametric problem instead, so estimation of E(Y|X=x), but not of
beta, still suffers the curse as M grows (page 11).

Closed-form rate arithmetic; no estimation, no randomness.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["dimredrate", "horowitz_dimension_reduction"]


def dimredrate(d, n, s=2, M=1):
    """Compare full-nonparametric, index and parametric rates.

    Parameters
    ----------
    d : int
        Dimension of X.
    n : int
        Sample size.
    s : int, default 2
        Smoothness / kernel order.
    M : int, default 1
        Number of indices; M = 1 is the single-index model.

    Returns
    -------
    RichResult
        payload keys: fullexp, fullrate, indexexp, indexrate, betaexp,
        betarate, gain, effdim, d, M, s, n, method.
    """
    d = int(d)
    n = int(n)
    s = int(s)
    M = int(M)
    if d < 1 or n < 1 or s < 1 or M < 1:
        raise ValueError("d, n, s and M must all be positive integers.")
    fullexp = s / float(2 * s + d)
    idxexp = s / float(2 * s + M)
    betaexp = 0.5
    fullrate = float(n ** (-fullexp))
    idxrate = float(n ** (-idxexp))
    return RichResult(
        title="Dimension reduction in an index model",
        payload={"fullexp": fullexp, "fullrate": fullrate,
                 "indexexp": idxexp, "indexrate": idxrate,
                 "betaexp": betaexp, "betarate": float(n ** -0.5),
                 "gain": float(fullrate / idxrate), "effdim": M,
                 "d": d, "M": M, "s": s, "n": n,
                 "method": "Horowitz (2009) Sections 1.2, 2.2, 2.4 rate comparison"},
    )


horowitz_dimension_reduction = dimredrate


def cheatsheet():
    return "hrzdimr: index models reduce a d-dimensional rate to an M-dimensional one"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    r = dimredrate(5, 10000)
    assert abs(r["indexexp"] - 2.0 / 5.0) < 1e-12, r["indexexp"]
    assert abs(r["fullexp"] - 2.0 / 9.0) < 1e-12, r["fullexp"]
    assert r["indexrate"] < r["fullrate"]      # the index model is faster
    assert r["betarate"] < r["indexrate"]      # beta is faster still
    assert abs(dimredrate(5, 10000, M=5)["indexexp"] - r["fullexp"]) < 1e-12
    print("ok", r["fullrate"], r["indexrate"], r["betarate"])
