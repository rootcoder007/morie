# morie.fn -- function file (rootcoder007/morie)
"""Differentially private synthetic data."""

from __future__ import annotations

from . import _array_core as np

from ._dp import check_budget, clip_to_range
from ._richresult import RichResult

__all__ = ["dp_synthetic_data"]


def dp_synthetic_data(X, epsilon=1.0, n_synth=None, bins=10, bounds=None, seed=None):
    r"""Generate synthetic records from a privately released histogram.

    A per-feature histogram is released by the Laplace mechanism and sampled
    from. Because sampling is post-processing, the synthetic records inherit
    the same :math:`\varepsilon` no matter how many are generated -- a million
    synthetic rows cost exactly what one costs.

    That is the appeal and also the trap. Synthetic data preserves only what
    the released statistics captured; here that is the **marginals only**, so
    every correlation between features is destroyed. Any analysis of the
    synthetic data that depends on joint structure -- a regression, a
    correlation matrix, an interaction -- will be badly wrong while looking
    entirely plausible, because the marginals match.

    ``preserved`` names what survives, and ``destroyed`` names what does not.
    A synthetic dataset that does not state its own fidelity is worse than no
    synthetic dataset, since it invites exactly the analyses it cannot
    support.

    Parameters
    ----------
    X : array-like
        Real data ``(n, p)``.
    epsilon : float
        Privacy budget, covering all features by parallel composition.
    n_synth : int, optional
        Records to generate. Defaults to ``n``.
    bins : int
        Histogram bins per feature.
    bounds : tuple, optional
        ``(low, high)`` chosen independently of the data.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``synthetic``, ``preserved``, ``destroyed``,
        ``marginal_error``, ``correlation_real``, ``correlation_synthetic``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    Marginals are approximately preserved.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(size=3000)
    >>> X = np.column_stack([z, z + rng.normal(0, 0.2, 3000)])   # strongly correlated
    >>> r = dp_synthetic_data(X, epsilon=10.0, bounds=(-4, 4), seed=1)
    >>> bool(abs(float(r["synthetic"][:, 0].mean()) - float(X[:, 0].mean())) < 0.5)
    True

    Correlation is destroyed, which is the thing to know before using it.

    >>> bool(abs(r["correlation_real"]) > 0.9)
    True
    >>> bool(abs(r["correlation_synthetic"]) < 0.3)
    True

    Generating more records costs no extra privacy, since sampling is
    post-processing.

    >>> big = dp_synthetic_data(X, epsilon=10.0, n_synth=50000, bounds=(-4, 4), seed=1)
    >>> float(big["epsilon"]) == float(r["epsilon"])
    True

    >>> str(r["destroyed"][0])
    'all inter-feature correlation'
    """
    epsilon, _ = check_budget(epsilon)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    n_synth = int(n if n_synth is None else n_synth)
    warn = []
    if bounds is None:
        lo, hi = float(X.min()), float(X.max())
        warn.append("bounds were taken from the data, which is a non-private query")
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
    Xc, lo, hi = clip_to_range(X, lo, hi)

    rng = np.random.default_rng(seed)
    synth = np.empty((n_synth, p))
    marg_err = np.empty(p)
    for j in range(p):
        counts, edges = np.histogram(Xc[:, j], bins=bins, range=(lo, hi))
        noisy = np.maximum(counts + rng.laplace(0.0, 2.0 / epsilon, counts.size), 0.0)
        if noisy.sum() <= 0:
            noisy = np.ones_like(noisy)
        prob = noisy / noisy.sum()
        idx = rng.choice(counts.size, size=n_synth, p=prob)
        synth[:, j] = rng.uniform(edges[idx], edges[idx + 1])
        marg_err[j] = float(np.abs(prob - counts / max(counts.sum(), 1)).sum())

    cr = float(np.corrcoef(Xc[:, 0], Xc[:, 1])[0, 1]) if p >= 2 else float("nan")
    cs = float(np.corrcoef(synth[:, 0], synth[:, 1])[0, 1]) if p >= 2 else float("nan")
    return RichResult(
        title="DP synthetic data",
        summary_lines=[("epsilon", epsilon), ("records", n_synth),
                       ("features", p)],
        warnings=warn + ["this preserves MARGINALS ONLY; every inter-feature "
                         "correlation is destroyed, so regressions and "
                         "interactions on it will be wrong while looking "
                         "plausible"],
        payload={
            "synthetic": synth,
            "preserved": ["per-feature marginal distributions"],
            "destroyed": ["all inter-feature correlation",
                          "joint structure", "interactions"],
            "marginal_error": marg_err,
            "correlation_real": cr, "correlation_synthetic": cs,
            "epsilon": epsilon, "n_synth": n_synth, "bins": int(bins),
            "method": "dp_synthetic_data",
        },
    )


def cheatsheet():
    return "dpsyn: marginals only -- correlations are DESTROYED, so regressions on it are wrong but plausible"
