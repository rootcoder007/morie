"""Hajek IPW average treatment effect.

Mirrors R ``morie_estimate_ate``.  Distinct from :mod:`morie.fn.hajek`,
which is the SURVEY Hajek mean: this is a DIFFERENCE of two weighted
means, the stabilised-IPW contrast

    ATE = sum_{T=1} w y / sum_{T=1} w  -  sum_{T=0} w y / sum_{T=0} w,
    w = T / e(X) + (1 - T) / (1 - e(X)).

The influence-function standard error is the KNOWN-propensity form
(Hernan and Robins, "What If", Ch. 12.6); when the propensity is
estimated it is conservative, which is stated rather than hidden.
"""

import math

from . import _array_core as np
from ._richresult import RichResult
from .aipw import _ps_keep, _trim_ps, _trim_weights
from .ps_fit import compute_propensity_scores

__all__ = ["hajekate", "hajek_ipw_ate"]


def hajekate(data, treatment, outcome, covariates, propensity_col=None,
             trim=(0.01, 0.99), trim_type="value", ps_model="mle",
             ridge_lambda=1.0, weight_trim=None, weight_trim_side="upper"):
    """Hajek IPW ATE.  Arguments match R ``morie_estimate_ate``."""
    required = [treatment, outcome, *covariates]
    frame = data.loc[:, required].dropna()
    t = np.asarray([float(v) for v in frame[treatment]], dtype=float)
    y = np.asarray([float(v) for v in frame[outcome]], dtype=float)
    if propensity_col is not None:
        ps = _trim_ps([float(v) for v in data[propensity_col]], trim, trim_type)
    else:
        ps = _trim_ps(compute_propensity_scores(
            frame, treatment=treatment, covariates=covariates,
            ps_model=ps_model, ridge_lambda=ridge_lambda).values,
            trim, trim_type)
    keep = _ps_keep(ps, trim, trim_type)
    n_discarded = int(sum(1 for k in keep if not k))
    if n_discarded:
        idx = [i for i, k in enumerate(keep) if k]
        if len(idx) < 2:
            raise ValueError("discard trimming removed almost every unit")
        t = np.asarray([t[i] for i in idx], dtype=float)
        y = np.asarray([y[i] for i in idx], dtype=float)
        ps = np.asarray([ps[i] for i in idx], dtype=float)
    w = np.asarray([t[i] / ps[i] + (1.0 - t[i]) / (1.0 - ps[i])
                    for i in range(len(t))], dtype=float)
    w = _trim_weights(w, weight_trim, weight_trim_side)
    i1 = [i for i in range(len(t)) if t[i] == 1.0]
    i0 = [i for i in range(len(t)) if t[i] == 0.0]
    m1 = sum(w[i] * y[i] for i in i1) / sum(w[i] for i in i1)
    m0 = sum(w[i] * y[i] for i in i0) / sum(w[i] for i in i0)
    ate = m1 - m0
    inf = [t[i] * y[i] / ps[i] - (1.0 - t[i]) * y[i] / (1.0 - ps[i]) - ate
           for i in range(len(t))]
    n = len(y)
    mu = sum(inf) / n
    se = math.sqrt(sum((v - mu) ** 2 for v in inf) / (n - 1)) / math.sqrt(n)
    z = 1.959963984540054
    sw = sum(w)
    return RichResult(payload={
        "estimate": ate, "ate": ate, "se": se,
        "ci_lower": ate - z * se, "ci_upper": ate + z * se,
        "ess": (sw * sw) / sum(v * v for v in w), "n": n,
        "n_discarded": n_discarded,
        "estimand": ("ATE on the retained subpopulation (Crump et al. "
                     "2009 discard trimming)" if n_discarded else
                     "ATE on the full sample"),
        "method": "Hajek IPW ATE (stabilised weights); known-propensity IF SE",
    })


hajek_ipw_ate = hajekate


def cheatsheet():
    return ("hajekate: Hajek IPW ATE = difference of weighted means; "
            "mirrors R morie_estimate_ate")
