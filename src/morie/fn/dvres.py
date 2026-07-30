# morie.fn -- function file (rootcoder007/morie)
"""Deviance residuals for a Cox model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .coxmgr import cox_martingale_residuals

__all__ = ["deviance_residual_cox"]


def deviance_residual_cox(fit):
    r"""Deviance residuals -- a symmetrised transform of the martingale ones.

    .. math::
        d_i = \operatorname{sign}(\hat M_i)
              \sqrt{-2\left[\hat M_i
                + \delta_i \log(\delta_i - \hat M_i)\right]} .

    The transform exists purely to fix the martingale residuals' left skew:
    deviance residuals are roughly symmetric about zero and, in large samples
    with light censoring, roughly standard normal. That makes the usual
    diagnostic conventions apply -- values beyond about :math:`\pm 2.5` are
    worth inspecting as poorly fitted subjects.

    The symmetry is only approximate, and it degrades as censoring gets heavy:
    with most subjects censored the residuals pile up just below zero however
    good the model is. Reading "too many negatives" as lack of fit under heavy
    censoring is a standard misdiagnosis.

    Parameters
    ----------
    fit : mapping
        A result from one of the Cox fitters.

    Returns
    -------
    RichResult
        ``residuals``, ``martingale``, ``n_extreme``, ``mean``, ``sd``.

    References
    ----------
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    Deviance residuals are far more symmetric than the martingale residuals
    they come from -- that is the entire purpose of the transform.

    >>> import numpy as np
    >>> from scipy.stats import skew
    >>> from morie.fn.efrnt import efron_tie_correction
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 2))
    >>> T = rng.exponential(1 / np.exp(X @ [0.8, -0.5]))
    >>> C = rng.exponential(3.0, 500)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> fit = efron_tie_correction(t, e, X)
    >>> r = deviance_residual_cox(fit)
    >>> bool(abs(skew(r["residuals"])) < abs(skew(r["martingale"])))
    True

    Sign is preserved from the martingale residual.

    >>> bool(np.array_equal(np.sign(r["residuals"]), np.sign(r["martingale"])))
    True
    """
    m = cox_martingale_residuals(fit)
    M = m["residuals"]
    e = m["event"]
    inner = -2.0 * (M + np.where(e > 0, e * np.log(np.maximum(e - M, 1e-300)), 0.0))
    d = np.sign(M) * np.sqrt(np.maximum(inner, 0.0))
    return RichResult(
        title="Cox deviance residuals",
        summary_lines=[("n", int(d.size)), ("mean", float(d.mean())),
                       ("sd", float(d.std(ddof=1)) if d.size > 1 else float("nan"))],
        warnings=["symmetry degrades under heavy censoring; a mass of small "
                  "negative residuals is expected there, not lack of fit"],
        payload={
            "residuals": d, "martingale": M,
            "n_extreme": int(np.sum(np.abs(d) > 2.5)),
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
            "method": "deviance_residual_cox",
        },
    )


def cheatsheet():
    return "dvres: symmetrised martingale residuals, ~N(0,1); symmetry fails under heavy censoring"
