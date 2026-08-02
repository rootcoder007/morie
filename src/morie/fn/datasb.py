# morie.fn -- function file (rootcoder007/morie)
"""Data subset refutation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["data_subset_refutation"]


def data_subset_refutation(estimator, y, d, X, fraction=0.8, n_sims=50,
                           seed=0, tol=0.1):
    r"""Re-estimate on random subsets; the estimate should be stable.

    Removing a random fraction of the rows removes no systematic
    information, so a causal estimate should move only by sampling
    noise. A LARGE shift means the estimate is being driven by a few
    observations -- typically ones with extreme propensity weights --
    rather than by the sample.

    The test is naturally compared against its own sampling error. On
    a fraction :math:`f` the subset standard deviation should be about
    :math:`\sqrt{1/f - 1}` times the estimator's own standard error, so
    ``excess_variability`` divides the observed spread by that
    expectation. A value near 1 means the variation is exactly what
    subsampling implies; well above 1 means particular rows matter.

    ``max_single_row_influence`` runs the complementary check
    directly: the largest change from deleting ONE observation.

    Parameters
    ----------
    estimator : callable
        ``estimator(y, d, X) -> float``.
    y, d, X : array-like
    fraction : float
        Share of rows retained.
    n_sims : int
    seed : int
    tol : float

    Returns
    -------
    RichResult
        ``original``, ``subset_mean``, ``subset_sd``,
        ``relative_change``, ``passed``, ``excess_variability``,
        ``max_single_row_influence``.

    References
    ----------
    Molak (2023), *Causal Inference and Discovery in Python*, chapter 7.
    Sharma and Kiciman (2020), DoWhy, arXiv:2011.04216.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn.drblr import doubly_robust_ate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2))
    >>> d = (rng.uniform(size=300) < 0.5).astype(float)
    >>> y = 2.0 * d + rng.normal(size=300)
    >>> f = lambda yy, dd, XX: doubly_robust_ate(yy, dd, XX)["estimate"]
    >>> bool(data_subset_refutation(f, y, d, X, n_sims=5)["passed"])
    True
    """
    if not callable(estimator):
        raise ValueError("estimator must be callable.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1), got %r." % fraction)
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    orig = float(estimator(yv, dv, Xa))
    rng = np.random.default_rng(int(seed))
    k = max(int(round(fraction * n)), 3)
    vals = []
    for _ in range(int(n_sims)):
        idx = rng.choice(n, size=k, replace=False)
        try:
            vals.append(float(estimator(yv[idx], dv[idx], Xa[idx])))
        except Exception:
            continue
    if not vals:
        raise ValueError("every subset failed to produce an estimate.")
    vals = np.asarray(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    denom = max(abs(orig), 1e-12)
    rel = float(abs(mean - orig) / denom)

    # leave-one-out influence, on a capped number of rows
    loo = []
    probe = rng.choice(n, size=min(n, 60), replace=False)
    for i in probe:
        m = np.ones(n, dtype=bool)
        m[i] = False
        try:
            loo.append(abs(float(estimator(yv[m], dv[m], Xa[m])) - orig))
        except Exception:
            continue
    infl = float(max(loo)) if loo else np.nan

    # expected subsample spread relative to the full-sample SE
    expected = np.sqrt(1.0 / fraction - 1.0)
    se_full = sd / expected if expected > 0 else np.nan
    return RichResult(
        payload={
            "estimate": mean,
            "original": orig,
            "subset_mean": mean,
            "subset_sd": sd,
            "subset_values": vals,
            "relative_change": rel,
            "passed": bool(rel < tol),
            "tolerance": float(tol),
            "fraction": float(fraction),
            "implied_se": se_full,
            "excess_variability": (float(sd / (expected * se_full))
                                   if se_full and se_full > 0 else np.nan),
            "variability_note": (
                "subset spread divided by what subsampling alone implies "
                "(sqrt(1/f - 1) times the full-sample SE); near 1 is normal, "
                "well above 1 means particular rows are driving the estimate"
            ),
            "max_single_row_influence": infl,
            "influence_note": (
                "largest change from deleting a single observation; a large "
                "value usually points at an extreme propensity weight"
            ),
            "n_sims": int(vals.size),
            "n": int(n),
            "method": "Data subset refutation",
        }
    )


def cheatsheet():
    return (
        "datasb: subset stability against the spread subsampling implies, "
        "plus leave-one-out influence"
    )
