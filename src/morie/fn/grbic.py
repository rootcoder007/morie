# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BIC for GMM model selection."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bic_gmm"]

_METHOD = "Bayesian information criterion"


def geron_bic_gmm(log_likelihood, n, n_params):
    r"""Bayesian information criterion; lower is better.

    .. math::
        \text{BIC} = p \log n - 2 \log \hat L

    The penalty grows with sample size, so BIC is stricter than AIC
    whenever :math:`\log n > 2`, i.e. for ``n > 7``.  On a GMM this
    usually means BIC picks fewer clusters.

    Parameters
    ----------
    log_likelihood : float
        Maximised log-likelihood of the fitted mixture.
    n : int
        Number of observations, at least 1.
    n_params : int
        Number of free parameters.

    Returns
    -------
    RichResult
        Payload keys ``bic``, ``log_likelihood``, ``n_params``,
        ``penalty``, ``penalty_per_param``, ``stricter_than_aic``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 8, BIC / AIC section.

    Examples
    --------
    >>> import math
    >>> r = geron_bic_gmm(-100.0, 100, 5)
    >>> round(r["bic"], 6) == round(5 * math.log(100) + 200.0, 6)
    True
    >>> round(float(r), 4)
    223.0259
    >>> r["stricter_than_aic"]
    True
    """
    log_likelihood = float(log_likelihood)
    if not np.isfinite(log_likelihood):
        raise ValueError(f"log_likelihood must be finite, got {log_likelihood}.")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1 observation, got {n}.")
    p = int(n_params)
    if p < 0:
        raise ValueError(f"n_params must be non-negative, got {p}.")

    per_param = float(np.log(n))
    penalty = per_param * p
    bic = penalty - 2.0 * log_likelihood

    return RichResult(
        title="BIC (GMM model selection)",
        summary_lines=[("BIC", bic), ("Penalty per parameter", per_param)],
        interpretation="Lower BIC is better; BIC penalises complexity harder than AIC once n > 7.",
        payload={
            "bic": bic,
            "log_likelihood": log_likelihood,
            "n_params": p,
            "penalty": penalty,
            "penalty_per_param": per_param,
            "stricter_than_aic": bool(per_param > 2.0),
            "estimate": bic,
            "n": n,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbic: BIC = log(n)*p - 2*logL for GMM model selection"


# compact alias per ledger/NAMING.md
geronbicgmm = geron_bic_gmm
