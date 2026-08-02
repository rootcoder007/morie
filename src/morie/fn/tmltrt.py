# morie.fn -- function file (rootcoder007/morie)
"""TMLE with propensity truncation."""

from . import _array_core as np

from ._richresult import RichResult
from ._tmle import tmle_ate

__all__ = ["tmle_truncation"]


def tmle_truncation(y, D, X, eps_grid=(0.001, 0.01, 0.025, 0.05, 0.1)):
    r"""TMLE across a grid of propensity truncation levels.

    Truncating :math:`g` to :math:`[\varepsilon, 1-\varepsilon]` bounds
    the clever covariate and hence the variance, at the cost of bias
    from the units whose scores are moved. Sweeping :math:`\varepsilon`
    exposes that trade-off directly: a stable plateau means positivity
    is fine, while an estimate that moves sharply with the truncation
    level is being driven by a handful of extreme weights and should
    not be reported as a point estimate.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    eps_grid : sequence of float, default (0.001, ..., 0.1)
        Truncation levels to sweep.

    Returns
    -------
    RichResult
        keys: ``eps`` (grid), ``ate`` (matching array), ``se``,
        ``n_truncated`` (units affected at each level), ``range``
        (max - min ATE across the grid), ``stable`` (range below one
        standard error), ``n``, ``method``.

    References
    ----------
    Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y. & van der
    Laan, M. J. (2012). Diagnosing and responding to violations in the
    positivity assumption. *Statistical Methods in Medical Research*,
    21(1), 31-54.
    """
    grid = np.asarray(eps_grid, dtype=float).ravel()
    if grid.size < 2 or np.any((grid <= 0) | (grid >= 0.5)):
        raise ValueError("eps_grid needs at least 2 values, all strictly inside (0, 0.5).")

    base = tmle_ate(y, D, X, trunc=1e-6)
    g0 = base["g"]

    ates, ses, ntr = [], [], []
    for e in grid:
        out = tmle_ate(y, D, X, trunc=float(e))
        ates.append(out["ate"])
        ses.append(out["se"])
        ntr.append(int(np.sum((g0 < e) | (g0 > 1 - e))))
    ates = np.array(ates)

    rng = float(ates.max() - ates.min())
    return RichResult(
        payload={
            "eps": grid,
            "ate": ates,
            "se": np.array(ses),
            "n_truncated": np.array(ntr),
            "range": rng,
            "stable": bool(rng < float(np.mean(ses))),
            "n": base["n"],
            "method": "TMLE truncation sweep (bias-variance trade-off diagnostic)",
        }
    )


def cheatsheet():
    return "tmltrt: sweep the g-truncation bound; a moving estimate = positivity trouble"
