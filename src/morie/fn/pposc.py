# morie.fn -- slice s03 (rootcoder007/morie)
"""Posterior predictive check and the Bayesian p-value.

Source consulted: Meng, X.-L. (1994).  Posterior predictive p-values.
*The Annals of Statistics* 22(3), 1142-1160, and Gelman, A., Meng, X.-L.
and Stern, H. (1996).  Posterior predictive assessment of model fitness
via realized discrepancies.  *Statistica Sinica* 6(4), 733-760.  The
p-value is

    p_B = Pr( T(y_rep, theta) >= T(y, theta) | y )

estimated as the proportion of replicated datasets whose discrepancy
exceeds the observed one.  Neither paper was retrievable here as a full
text; the definition is quoted in its standard published form.

The known property of p_B is that it is *conservative* -- its
distribution under the model is not uniform but concentrated around 1/2
-- so a value near 0.5 is evidence of nothing, and only extremes are
informative.  That is stated in ``interpretation`` rather than left for
the user to rediscover.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["posterior_predictive_check"]

_STATS = {
    "mean": lambda v: k.mean(v),
    "sd": lambda v: k.sd(v, 1),
    "min": lambda v: min(v),
    "max": lambda v: max(v),
    "median": lambda v: k.median(v),
}


def posterior_predictive_check(y, y_rep, statistic="mean"):
    """Bayesian p-value for a chosen test statistic.

    Parameters
    ----------
    y : array-like
        The observed data.
    y_rep : 2-D array-like
        Replicated datasets, one row per posterior draw.
    statistic : str or callable
        One of "mean", "sd", "min", "max", "median", or a callable.

    Returns
    -------
    estimate : the Bayesian p-value
    t_obs    : the observed statistic
    t_rep    : the statistic of every replicate
    """
    v = k.vec(y)
    R = k.mat(y_rep)
    f = statistic if callable(statistic) else _STATS.get(str(statistic),
                                                         _STATS["mean"])
    tobs = float(f(v))
    trep = [float(f(list(row))) for row in R]
    ge = 0.0
    for t in trep:
        if t >= tobs:
            ge += 1.0
    p = ge / len(trep) if trep else float("nan")
    return RichResult(
        title="Posterior predictive check",
        summary_lines=[("p_B", p), ("T(y)", tobs)],
        interpretation=("The posterior predictive p-value is conservative: "
                        "under the model its distribution is concentrated "
                        "around 1/2, so only extreme values are informative."),
        payload={
            "estimate": p,
            "p_value": p,
            "t_obs": tobs,
            "t_rep": trep,
            "mean_t_rep": k.mean(trep) if trep else float("nan"),
            "n_rep": len(trep),
            "method": "Posterior predictive p-value (Meng 1994; Gelman, Meng and Stern 1996)",
        },
    )


def cheatsheet():
    return "pposc: Posterior predictive check (Bayesian p-value)"
