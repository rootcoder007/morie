# morie.fn -- function file (rootcoder007/morie)
"""Warm's weighted likelihood ability estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["weighted_likelihood_theta"]


def weighted_likelihood_theta(y, a=None, b=None, c=None,
                              bounds=(-6.0, 6.0)):
    r"""Warm's (1989) weighted likelihood estimate: maximise

    .. math:: \log\ell(\theta) + \log\sqrt{I(\theta)},

    the likelihood weighted by the square root of the test
    information.

    The weight is not a prior and the distinction matters. Warm's
    derivation chooses it so that the :math:`O(1/n)` BIAS of the
    maximum-likelihood estimator is removed to first order -- the
    WLE is bias-corrected, where MAP and EAP are shrunk toward a
    prior mean they do not claim to be unbiased about. On a test
    with no prior information to bring, that makes the WLE the
    estimator of choice, and the tests check the bias reduction by
    simulation against ML across replications rather than asserting
    it.

    It also inherits the prior-free existence property in a limited
    but useful form: because :math:`\sqrt{I(\theta)} \to 0` in both
    tails, the weighted objective turns over where the raw
    likelihood does not, so a FINITE estimate exists for perfect
    patterns too -- the one property MAP buys with a prior, obtained
    here without one.

    Parameters
    ----------
    y : array-like of 0/1
        Item responses.
    a, b, c : array-like, optional
        Item parameters; ``b`` required.
    bounds : (float, float)
        Search interval.

    Returns
    -------
    RichResult
        keys: ``theta``, ``se``, ``information``, ``loglik``,
        ``weight_term``, ``bias_corrected`` (True),
        ``finite_for_perfect_patterns`` (True), ``vs_ml``,
        ``n_items``, ``method``.

    References
    ----------
    Warm, T. A. (1989), "Weighted likelihood estimation of ability
    in item response theory", *Psychometrika* 54:427-450.
    """
    from ._psycho import logistic_3pl, logistic_3pl_deriv

    yv = np.asarray(y, dtype=float).ravel()
    m = yv.size
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("responses must be binary 0/1.")
    if b is None:
        raise ValueError("item difficulties b are required.")
    bv = np.asarray(b, dtype=float).ravel()
    av = np.ones(m) if a is None else np.asarray(a, dtype=float).ravel()
    cv = np.zeros(m) if c is None else np.asarray(c, dtype=float).ravel()
    if not (bv.size == av.size == cv.size == m):
        raise ValueError("a, b, c must each have one entry per item.")
    grid = np.linspace(float(bounds[0]), float(bounds[1]), 8001)
    P = np.clip(logistic_3pl(grid, av, bv, cv), 1e-12, 1 - 1e-12)
    dP = logistic_3pl_deriv(grid, av, bv, cv)
    info = np.sum(dP ** 2 / (P * (1 - P)), axis=1)
    ll = (yv * np.log(P) + (1 - yv) * np.log(1 - P)).sum(axis=1)
    obj = ll + 0.5 * np.log(np.maximum(info, 1e-300))
    i = int(np.argmax(obj))
    th = float(grid[i])
    if 0 < i < grid.size - 1:
        y0, y1, y2 = obj[i - 1], obj[i], obj[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            th = float(grid[i] - 0.5 * (grid[1] - grid[0])
                       * (y2 - y0) / den)
    Pt = np.clip(logistic_3pl(np.array([th]), av, bv, cv)[0],
                 1e-12, 1 - 1e-12)
    dPt = logistic_3pl_deriv(np.array([th]), av, bv, cv)[0]
    info_t = float(np.sum(dPt ** 2 / (Pt * (1 - Pt))))
    ml_theta = None
    try:
        from .mleth import mle_theta_estimator
        ml = mle_theta_estimator(y, a=av, b=bv, c=cv, bounds=bounds)
        ml_theta = ml["theta"] if ml["finite"] else None
    except Exception:
        pass
    return RichResult(payload={
        "theta": th, "se": float(1 / np.sqrt(info_t)) if info_t > 0
        else np.inf,
        "information": info_t,
        "loglik": float(np.interp(th, grid, ll)),
        "weight_term": float(0.5 * np.log(max(info_t, 1e-300))),
        "bias_corrected": True,
        "finite_for_perfect_patterns": True,
        "why_finite": "sqrt(I(theta)) tends to zero in both tails, so the "
                      "weighted objective turns over where the raw "
                      "likelihood does not -- the existence property MAP "
                      "buys with a prior, obtained without one",
        "not_a_prior": "the weight removes the O(1/n) bias of the ML "
                       "estimator to first order (Warm 1989); it is a "
                       "bias correction, not prior information",
        "vs_ml": None if ml_theta is None else float(th - ml_theta),
        "n_items": int(m),
        "method": "Warm (1989) weighted likelihood: loglik + log sqrt(information)"})


def cheatsheet():
    return "wleth: sqrt(I) weight removes ML's O(1/n) bias AND makes perfect patterns finite -- no prior"
