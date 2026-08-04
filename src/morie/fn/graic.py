# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AIC for GMM model selection."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_aic_gmm", "gmm_n_params"]

_METHOD = "Akaike information criterion"


def gmm_n_params(k, d, covariance_type="full"):
    """Free-parameter count of a ``k``-component, ``d``-dimensional GMM.

    Weights contribute ``k - 1`` (they sum to one), means ``k*d``, and
    the covariances depend on the parameterisation.

    Examples
    --------
    >>> gmm_n_params(3, 2, "full")
    17
    >>> gmm_n_params(3, 2, "diag")
    14
    >>> gmm_n_params(3, 2, "spherical")
    11
    """
    k = int(k)
    d = int(d)
    if k < 1 or d < 1:
        raise ValueError(f"k and d must be >= 1, got k={k}, d={d}.")
    per_cov = {
        "full": d * (d + 1) // 2,
        "diag": d,
        "spherical": 1,
    }
    if covariance_type not in per_cov:
        raise ValueError(
            f"covariance_type must be one of {sorted(per_cov)}, got {covariance_type!r}."
        )
    return (k - 1) + k * d + k * per_cov[covariance_type]


def geron_aic_gmm(log_likelihood, n_params):
    r"""Akaike information criterion; lower is better.

    .. math::
        \text{AIC} = 2p - 2\log \hat L

    AIC penalises each free parameter by a flat 2, independent of sample
    size, so it tends to select richer mixtures than BIC on large data.

    Parameters
    ----------
    log_likelihood : float
        Maximised log-likelihood of the fitted mixture (a *log*, so
        typically negative).
    n_params : int
        Number of free parameters -- see :func:`gmm_n_params`.

    Returns
    -------
    RichResult
        Payload keys ``aic``, ``log_likelihood``, ``n_params``,
        ``penalty``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 8, BIC / AIC section.

    Examples
    --------
    >>> round(float(geron_aic_gmm(-100.0, 5)), 6)
    210.0
    """
    log_likelihood = float(log_likelihood)
    if not np.isfinite(log_likelihood):
        raise ValueError(f"log_likelihood must be finite, got {log_likelihood}.")
    p = int(n_params)
    if p < 0:
        raise ValueError(f"n_params must be non-negative, got {p}.")
    penalty = 2.0 * p
    aic = penalty - 2.0 * log_likelihood

    return RichResult(
        title="AIC (GMM model selection)",
        summary_lines=[("AIC", aic), ("Free parameters", p)],
        interpretation="Lower AIC is better; differences below ~2 are not decisive.",
        payload={
            "aic": aic,
            "log_likelihood": log_likelihood,
            "n_params": p,
            "penalty": penalty,
            "estimate": aic,
            "n": p,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "graic: AIC = 2p - 2*logL for GMM model selection"


# compact alias per ledger/NAMING.md
geronaicgmm = geron_aic_gmm


# compact alias per ledger/NAMING.md
gmmnparams = gmm_n_params
