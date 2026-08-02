# morie.fn -- function file (rootcoder007/morie)
"""Maximum-likelihood ability estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mle_theta_estimator"]


def mle_theta_estimator(y, a=None, b=None, c=None, bounds=(-6.0, 6.0)):
    r"""Maximum-likelihood estimate of the latent trait
    :math:`\theta` under the 3PL model, maximising

    .. math:: \ell(\theta) = \sum_j \left[y_j \log P_j(\theta)
              + (1-y_j)\log\{1 - P_j(\theta)\}\right].

    **The estimate does not exist for an all-correct or all-wrong
    response pattern.** The likelihood is then monotone in
    :math:`\theta` and its supremum sits at :math:`\pm\infty`; there
    is no maximum to find. This module says so -- ``finite`` is
    ``False`` and ``theta`` is :math:`\pm\infty` -- rather than
    returning the edge of a search interval as though it were an
    estimate, which is the standard silent failure and the reason
    :func:`morie.fn.mapth.map_theta_estimator` and
    :func:`morie.fn.wleth.weighted_likelihood_theta` exist.

    Under the 3PL the likelihood can also be MULTIMODAL when
    guessing is present (Samejima 1973; Yen, Burket and Sykes 1991),
    so the maximisation is a dense scan followed by local
    refinement rather than a Newton step from an arbitrary start,
    and ``n_local_maxima`` reports what was found.

    The standard error is :math:`1/\sqrt{I(\hat\theta)}` with the
    test information
    :math:`I(\theta) = \sum_j P_j'^2 / \{P_j(1-P_j)\}`.

    Parameters
    ----------
    y : array-like of 0/1
        Item responses.
    a, b, c : array-like, optional
        Discrimination, difficulty, guessing. Defaults: ``a = 1``,
        ``c = 0`` (giving the Rasch/2PL cases), ``b`` required.
    bounds : (float, float)
        Search interval for the scan.

    Returns
    -------
    RichResult
        keys: ``theta``, ``se``, ``finite``, ``information``,
        ``loglik``, ``n_local_maxima``, ``pattern``, ``n_items``,
        ``why_infinite``, ``method``.

    References
    ----------
    Lord, F. M. (1980), *Applications of Item Response Theory to
    Practical Testing Problems*, Erlbaum, Ch. 4. Samejima, F.
    (1973), *Psychometrika* 38:221-233, and Yen, Burket and Sykes
    (1991), *Psychometrika* 56:39-54, for multimodality.
    """
    from ._psycho import logistic_3pl, logistic_3pl_deriv

    yv = np.asarray(y, dtype=float).ravel()
    m = yv.size
    if m < 1:
        raise ValueError("need at least one item.")
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("responses must be binary 0/1.")
    if b is None:
        raise ValueError("item difficulties b are required.")
    bv = np.asarray(b, dtype=float).ravel()
    av = np.ones(m) if a is None else np.asarray(a, dtype=float).ravel()
    cv = np.zeros(m) if c is None else np.asarray(c, dtype=float).ravel()
    if not (bv.size == av.size == cv.size == m):
        raise ValueError("a, b, c must each have one entry per item.")
    if np.any(cv < 0) or np.any(cv >= 1):
        raise ValueError("guessing parameters must lie in [0, 1).")

    allc = bool(np.all(yv == 1))
    allw = bool(np.all(yv == 0))
    if allc or allw:
        inf = np.inf if allc else -np.inf
        return RichResult(payload={
            "theta": inf, "se": np.inf, "finite": False,
            "information": 0.0, "loglik": np.nan,
            "n_local_maxima": 0,
            "pattern": "all correct" if allc else "all incorrect",
            "n_items": int(m),
            "why_infinite": "the likelihood is monotone in theta for a "
                            "perfect pattern, so its supremum is at "
                            "infinity and no maximum exists; use a MAP/EAP "
                            "estimator or Warm's weighted likelihood",
            "method": "3PL maximum likelihood (no finite maximum here)"})

    lo, hi = float(bounds[0]), float(bounds[1])
    grid = np.linspace(lo, hi, 4001)
    P = logistic_3pl(grid, av, bv, cv)
    P = np.clip(P, 1e-12, 1 - 1e-12)
    ll = (yv * np.log(P) + (1 - yv) * np.log(1 - P)).sum(axis=1)
    interior = np.flatnonzero((ll[1:-1] > ll[:-2]) & (ll[1:-1] >= ll[2:])) + 1
    i = int(np.argmax(ll))
    # local refinement by golden section around the winning node
    step = grid[1] - grid[0]
    left, right = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    phi = (np.sqrt(5) - 1) / 2

    def negll(t):
        p = np.clip(logistic_3pl(np.array([t]), av, bv, cv)[0],
                    1e-12, 1 - 1e-12)
        return -float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))

    x1, x2 = right - phi * (right - left), left + phi * (right - left)
    f1, f2 = negll(x1), negll(x2)
    for _ in range(80):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - phi * (right - left)
            f1 = negll(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + phi * (right - left)
            f2 = negll(x2)
        if right - left < 1e-10:
            break
    th = float((left + right) / 2)
    Pt = np.clip(logistic_3pl(np.array([th]), av, bv, cv)[0],
                 1e-12, 1 - 1e-12)
    dP = logistic_3pl_deriv(np.array([th]), av, bv, cv)[0]
    info = float(np.sum(dP ** 2 / (Pt * (1 - Pt))))
    return RichResult(payload={
        "theta": th, "se": float(1 / np.sqrt(info)) if info > 0 else np.inf,
        "finite": True, "information": info,
        "loglik": -negll(th),
        "n_local_maxima": int(interior.size),
        "multimodality_note": "the 3PL likelihood can be multimodal when "
                              "guessing is present (Samejima 1973; Yen et "
                              "al. 1991), so a dense scan precedes local "
                              "refinement rather than a Newton step",
        "pattern": "mixed", "n_items": int(m), "grid_step": float(step),
        "method": "3PL maximum-likelihood theta by scan plus golden-section refinement"})


def cheatsheet():
    return "mleth: no finite MLE for a perfect pattern -- says so instead of returning the search bound"
