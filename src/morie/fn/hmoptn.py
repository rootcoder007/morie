# morie.fn -- function file (rootcoder007/morie)
"""Tree-structured Parzen estimator for hyperparameter search."""

import numpy as np

from ._richresult import RichResult

__all__ = ["tpe_suggest", "geron_optuna"]


def tpe_suggest(trials, bounds, n_candidates=64, gamma=0.25, seed=0,
                bandwidth=None):
    r"""Suggest the next hyperparameter point by the TPE rule.

    TPE splits the observed trials at the :math:`\gamma` quantile of
    the objective and fits two densities: :math:`\ell(x)` over the good
    trials and :math:`g(x)` over the rest. The next point maximises

    .. math:: \frac{\ell(x)}{g(x)},

    which Bergstra et al. show is monotone in Expected Improvement. The
    inversion is the whole idea: instead of modelling
    :math:`p(y \mid x)` as a Gaussian process does, TPE models
    :math:`p(x \mid y)`, which is why it handles discrete and
    conditional search spaces a GP cannot.

    Random search is the honest baseline here, not grid search. Grid
    search wastes trials on dimensions that do not matter -- with
    :math:`d` parameters of which only one is influential, a grid
    spends :math:`n^{d-1}` evaluations re-testing the same value of
    that one. ``improvement_over_random`` reports the density ratio at
    the suggestion, which is how much the model thinks it has learned;
    near 1 means it has learned nothing yet and the suggestion is
    effectively random.

    Parameters
    ----------
    trials : sequence of (params, objective)
        ``params`` is a length-p sequence; lower ``objective`` is better.
    bounds : sequence of (low, high), length p
    n_candidates : int
        Candidates drawn from ``l`` per suggestion.
    gamma : float
        Quantile splitting good from bad.
    seed : int
    bandwidth : float, optional
        Kernel width as a fraction of each range. Scott's rule by
        default.

    Returns
    -------
    RichResult
        ``suggestion``, ``ratio``, ``n_good``, ``n_bad``,
        ``improvement_over_random``, ``best_so_far``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 10,
    hyperparameter tuning. Bergstra, Bardenet, Bengio and Kegl (2011),
    "Algorithms for hyper-parameter optimization", NeurIPS.
    Bergstra and Bengio (2012), *JMLR* 13:281-305, on random versus
    grid search.

    Examples
    --------
    >>> t = [([0.1], 1.0), ([0.9], 5.0), ([0.2], 1.2), ([0.8], 4.0)]
    >>> out = tpe_suggest(t, [(0.0, 1.0)], seed=1)
    >>> bool(0.0 <= out["suggestion"][0] <= 1.0)
    True
    """
    if not trials:
        raise ValueError("need at least one trial.")
    X = np.array([np.asarray(p, dtype=float).ravel() for p, _ in trials])
    y = np.array([float(v) for _, v in trials])
    n, p = X.shape
    B = np.atleast_2d(np.asarray(bounds, dtype=float))
    if B.shape != (p, 2):
        raise ValueError(
            "bounds must have one (low, high) pair per parameter: expected "
            "%d, got %s." % (p, B.shape)
        )
    if np.any(B[:, 1] <= B[:, 0]):
        raise ValueError("every bound must have high > low.")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1), got %r." % gamma)

    rng = np.random.default_rng(int(seed))
    span = B[:, 1] - B[:, 0]
    n_good = max(int(np.ceil(gamma * n)), 1)
    order = np.argsort(y)
    good, bad = X[order[:n_good]], X[order[n_good:]]
    if bad.shape[0] == 0:
        bad = X

    bw = (np.maximum(span * n ** (-1.0 / (p + 4)), 1e-9)
          if bandwidth is None
          else np.full(p, float(bandwidth)) * span)

    def dens(pts, sample):
        d = (sample[:, None, :] - pts[None, :, :]) / bw[None, None, :]
        k = np.exp(-0.5 * np.sum(d ** 2, axis=2))
        return k.mean(axis=1) + 1e-300

    pick = good[rng.integers(0, good.shape[0], size=int(n_candidates))]
    cand = np.clip(pick + rng.normal(size=(int(n_candidates), p)) * bw,
                   B[:, 0], B[:, 1])
    ratio = dens(good, cand) / dens(bad, cand)
    best = int(np.argmax(ratio))
    return RichResult(
        payload={
            "estimate": cand[best],
            "suggestion": cand[best],
            "ratio": float(ratio[best]),
            "candidates": cand,
            "candidate_ratios": ratio,
            "n_good": int(good.shape[0]),
            "n_bad": int(bad.shape[0]),
            "gamma": float(gamma),
            "bandwidth": bw,
            "improvement_over_random": float(ratio[best] / max(
                float(np.median(ratio)), 1e-300)),
            "improvement_note": (
                "the l/g ratio at the suggestion; near 1 means the model has "
                "learned nothing yet and the suggestion is effectively random"
            ),
            "best_so_far": float(y.min()),
            "best_params": X[int(np.argmin(y))],
            "tpe_note": (
                "TPE models p(x | y) rather than p(y | x), which is why it "
                "handles discrete and conditional spaces that a Gaussian "
                "process cannot"
            ),
            "n_trials": int(n),
            "method": "Tree-structured Parzen estimator suggestion",
        }
    )


def cheatsheet():
    return (
        "hmoptn: TPE next-point suggestion from the l/g density ratio, with "
        "how much it has actually learned"
    )


#: Catalogue alias for :func:`tpe_suggest`.
geron_optuna = tpe_suggest
