# morie.fn -- function file (rootcoder007/morie)
"""W-NOMINATE log-likelihood of a roll-call matrix."""

import numpy as np

from ._richresult import RichResult
from .wnomp import wnominate_probability

__all__ = ["wnominate_logit"]


def wnominate_logit(votes, ideal_points, yea_nay_positions, beta=15.0):
    r"""Bernoulli log-likelihood under the NOMINATE choice probabilities.

    .. math:: \log L = \sum_i \sum_j \big[ y_{ij} \log P_{ij}
              + (1 - y_{ij}) \log (1 - P_{ij}) \big],

    with :math:`P_{ij}` from
    :func:`morie.fn.wnomp.wnominate_probability` and missing votes
    (NaN) skipped. Also reports the three Armstrong Sec 5.3.5 fit
    statistics computed from the same probabilities: correct
    classification, APRE, and GMP = e^{log L / N}.

    Parameters
    ----------
    votes : array-like, shape (n, q)
        Binary roll-call matrix (1 = yea, 0 = nay, NaN = missing).
    ideal_points : array-like, shape (n, k)
        Legislator coordinates.
    yea_nay_positions : array-like, shape (q, 2, k)
        Yea and Nay outcome coordinates per roll call.
    beta : float, default 15.0

    Returns
    -------
    RichResult
        keys: ``loglik``, ``gmp``, ``correct_classification``,
        ``apre``, ``n_choices``, ``method``.

    References
    ----------
    Poole, K. T. & Rosenthal, H. (1997). *Congress*. Oxford
    University Press.

    Armstrong, D. A. et al. (2014). Sec. 5.3.5 footnote, p. 143 (the
    fit statistics).
    """
    V = np.asarray(votes, dtype=float)
    X = np.asarray(ideal_points, dtype=float)
    Z = np.asarray(yea_nay_positions, dtype=float)
    if V.ndim != 2:
        raise ValueError("votes must be 2-D.")
    n, q = V.shape
    if X.ndim != 2 or X.shape[0] != n:
        raise ValueError("ideal_points must be (n, k).")
    if Z.shape[:2] != (q, 2) or Z.shape[2] != X.shape[1]:
        raise ValueError("yea_nay_positions must be (q, 2, k).")

    ll = 0.0
    n_choices = 0
    correct = 0
    null_err = 0
    model_err = 0
    for j in range(q):
        valid = ~np.isnan(V[:, j])
        if not valid.any():
            continue
        p = wnominate_probability(X[valid], Z[j, 0], Z[j, 1], beta=beta)["p_yea"]
        p = np.clip(np.atleast_1d(p), 1e-10, 1 - 1e-10)
        y = V[valid, j]
        ll += float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
        n_choices += int(valid.sum())
        pred = (p > 0.5).astype(float)
        correct += int((pred == y).sum())
        yea = y.sum()
        minority = int(min(yea, y.size - yea))
        null_err += minority
        model_err += int((pred != y).sum())

    if n_choices == 0:
        raise ValueError("no non-missing votes.")
    apre = float((null_err - model_err) / null_err) if null_err > 0 else 1.0

    return RichResult(
        payload={
            "loglik": ll,
            "gmp": float(np.exp(ll / n_choices)),
            "correct_classification": correct / n_choices,
            "apre": apre,
            "n_choices": int(n_choices),
            "method": "W-NOMINATE Bernoulli log-likelihood + Sec 5.3.5 fit statistics",
        }
    )


def cheatsheet():
    return "wnoml: sum y log P + (1-y) log(1-P); GMP = exp(logL/N)"
