# morie.fn -- function file (rootcoder007/morie)
"""Pathwise derivative of a parameter."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch3_pathwise_derivative"]


def kosorok_ch3_pathwise_derivative(psi_values, scores, weights=None):
    r"""Pathwise differentiability of a parameter (Kosorok Ch. 3):

    .. math:: \frac{d}{dt}\psi(P_t)\Big|_{t=0} = a
              = P\big[\tilde\psi_{\theta,\eta}\,
              (\dot\ell_{\theta,\eta} a + g)\big].

    The parameter's derivative along a submodel equals the inner
    product of its influence function with the submodel's score. That
    identity is the whole content of "pathwise differentiable": the
    derivative is REPRESENTED by an influence function, which is what
    makes an efficiency bound meaningful.

    Given the influence-function values and score values on a sample,
    returns the empirical inner product and checks the two conditions
    an influence function must satisfy: mean zero, and finite
    variance.

    Parameters
    ----------
    psi_values : array-like, shape (n,)
        Influence-function values.
    scores : array-like, shape (n,) or (n, k)
        Score values along the submodel(s).
    weights : array-like, optional
        Observation weights.

    Returns
    -------
    RichResult
        keys: ``derivative`` (per submodel), ``influence_mean``,
        ``influence_var``, ``mean_zero`` (bool), ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 3 (pathwise differentiability).
    """
    psi = np.asarray(psi_values, dtype=float).ravel()
    S = np.asarray(scores, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    n = psi.size
    if S.shape[0] != n:
        raise ValueError("scores must have one row per influence value.")
    w = np.ones(n) / n if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError("weights must match the sample size.")
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    mean = float(w @ psi)
    var = float(w @ (psi - mean) ** 2)
    return RichResult(
        payload={"derivative": (w[:, None] * psi[:, None] * S).sum(axis=0),
                 "influence_mean": mean, "influence_var": var,
                 "mean_zero": bool(abs(mean) < 1e-6 * max(1.0, np.sqrt(var))),
                 "n": int(n),
                 "method": "d psi(P_t)/dt = P[psi-tilde * score]; the representation IS the point"}
    )


def cheatsheet():
    return "ksr062: derivative represented by an influence function; mean zero checked"
