# morie.fn -- function file (rootcoder007/morie)
"""Nonresponse adjustment via response propensity."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["nonresponse_adjustment"]


def nonresponse_adjustment(y, weights, propensity):
    r"""Adjust design weights by the inverse response propensity.

    Each respondent's design weight is divided by its estimated
    response probability, :math:`w_i^{adj} = w_i / \hat\varphi_i`, and
    the population mean is estimated by the Hajek ratio

    .. math:: \hat{\bar Y} = \frac{\sum_i w_i^{adj} y_i}
              {\sum_i w_i^{adj}}.

    Parameters
    ----------
    y : array-like, shape (r,)
        Respondent outcomes.
    weights : array-like, shape (r,)
        Respondent design weights (positive).
    propensity : array-like, shape (r,)
        Estimated response propensities in (0, 1].

    Returns
    -------
    RichResult
        keys: ``estimate`` (Hajek weighted mean), ``se`` (weighted
        ratio-estimator SE), ``weights_adjusted``, ``ess``, ``n``,
        ``method``.

    References
    ----------
    Little, R. J. A. & Rubin, D. B. (2002). *Statistical Analysis with
    Missing Data* (2nd ed.). Wiley. Sec. 3.3 (weighting adjustments).

    Little, R. J. & Vartivarian, S. (2005). Does weighting for
    nonresponse increase the variance of survey means? *Survey
    Methodology*, 31(2), 161-168.
    """
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    phi = np.asarray(propensity, dtype=float).ravel()
    if not (y.size == w.size == phi.size):
        raise ValueError("y, weights, propensity must have equal length.")
    if np.any(w <= 0):
        raise ValueError("weights must be positive.")
    if np.any((phi <= 0) | (phi > 1)):
        raise ValueError("propensity must lie in (0, 1].")

    wa = w / phi
    sw = wa.sum()
    est = float((wa * y).sum() / sw)
    # linearised SE of the Hajek ratio estimator
    n = y.size
    resid = wa * (y - est)
    se = float(np.sqrt(n / (n - 1) * (resid**2).sum()) / sw) if n > 1 else float("nan")
    ess = float(sw * sw / (wa**2).sum())

    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "weights_adjusted": wa,
            "ess": ess,
            "n": int(n),
            "method": "Nonresponse adjustment via response propensity",
        }
    )


def cheatsheet():
    return "nonresp: Hajek mean with weights w/phi (inverse response propensity)"
