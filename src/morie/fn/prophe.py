# morie.fn -- function file (rootcoder007/morie)
r"""Prophet's additive decomposition: the components, separated.

Same model and same source as :mod:`morie.fn.prphet` -- the ledger
carries both rows against Taylor & Letham (2018) and they describe one
method, so this module does not re-derive it. What it adds is the
*decomposition view*: fit the model once, then return each term of

.. math:: y(t) = g(t) + s(t) + h(t) + \epsilon_t

separately, which is what makes the model worth using. An analyst who
can see that a January dip is seasonal rather than a trend break can act
on it; a single fitted curve tells them nothing.

**The decomposition must add up, exactly.** Trend plus every
seasonality plus holidays must reconstruct the fitted values to machine
precision -- if it does not, a component has been dropped or
double-counted, and the plot an analyst reads will be wrong in a way no
error metric reveals. The anchor checks the sum, and checks that
removing a component changes the reconstruction.

**Contribution shares are a ranking, not a variance decomposition.**
The components are not orthogonal -- a Fourier term and a piecewise
trend can both absorb a slow drift -- so the share of variance each
explains does not partition. What is reported is each component's own
standard deviation over the sample, which ranks them honestly without
implying an additive split.

References
----------
Taylor, S. J. & Letham, B. (2018) "Forecasting at Scale", *The American
Statistician* 72(1), 37-45, doi:10.1080/00031305.2017.1380080;
preprint *PeerJ Preprints* 5:e3190v2,
doi:10.7287/peerj.preprints.3190v2. Eq. (1); the decomposition is its
Sec. 3.

Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I.
(1990) "STL: A Seasonal-Trend Decomposition Procedure Based on Loess",
*Journal of Official Statistics* 6(1), 3-73. The decomposition idea in
its nonparametric form.

Harvey, A. C. & Peters, S. (1990) "Estimation procedures for structural
time series models", *Journal of Forecasting* 9(2), 89-108,
doi:10.1002/for.3980090203.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .prphet import (fourier_terms, holiday_matrix, piecewise_trend,
                     prophet_fit)

__all__ = ["additive_components", "component_shares"]

_EPS = 1e-12


def additive_components(t, y, seasonalities=None, holidays=None,
                        holiday_window=(0, 0), **kw):
    r"""Fit once, then return :math:`g`, each :math:`s`, and :math:`h`.

    The components are returned separately AND their sum is checked
    against the fitted values, because a decomposition that does not
    reconstruct is a picture of something else.
    """
    fit = prophet_fit(t, y, seasonalities=seasonalities,
                      holidays=holidays,
                      holiday_window=holiday_window, **kw)
    tv = fit["t"]
    n = len(tv)
    coef = fit["coef"]
    trend = fit["trend"]

    comps = {"trend": trend}
    for (name, period, order) in (seasonalities or []):
        F = fourier_terms(tv, period, order)
        vals = []
        for i in range(n):
            s = 0.0
            for nn in range(1, int(order) + 1):
                s += (coef["%s_cos%d" % (name, nn)] * F[i][2 * nn - 2]
                      + coef["%s_sin%d" % (name, nn)] * F[i][2 * nn - 1])
            vals.append(s)
        comps[name] = vals
    if holidays:
        H, names = holiday_matrix(tv, holidays, holiday_window[0],
                                  holiday_window[1])
        comps["holidays"] = [
            sum(coef["holiday_%s" % names[j]] * H[i][j]
                for j in range(len(names))) for i in range(n)]

    total = [sum(comps[c][i] for c in comps) for i in range(n)]
    gap = max(abs(total[i] - fit["fitted"][i]) for i in range(n))
    return RichResult(payload={
        "estimate": comps, "components": comps, "total": total,
        "fitted": fit["fitted"], "residual": fit["residual"],
        "reconstruction_error": gap, "reconstructs": gap < 1e-8,
        "coef": coef, "changepoints": fit["changepoints"],
        "sigma": fit["sigma"], "n": n,
        "component_names": sorted(comps),
        "method": "Prophet additive decomposition, Taylor & Letham "
                  "(2018) eq. (1)",
    })


def component_shares(components):
    r"""Each component's own standard deviation, as a ranking.

    NOT a variance decomposition: the components are not orthogonal, so
    these do not partition the variance and are not presented as if
    they did.
    """
    out = {}
    for name, vals in components.items():
        out[name] = k.sd(vals) if len(vals) > 1 else 0.0
    tot = sum(out.values())
    return {"sd": out,
            "relative": ({n: v / tot for n, v in out.items()}
                         if tot > 0 else {n: 0.0 for n in out}),
            "ranked": sorted(out, key=lambda n: -out[n]),
            "note": "standard deviations, not an orthogonal variance "
                    "split -- the components overlap"}


def cheatsheet():
    return ("prophe: same model and source as prphet (Taylor & Letham "
            "2018 eq. 1) -- this is the DECOMPOSITION view. Fit once, "
            "return g(t), each s(t) and h(t) separately, and check they "
            "sum back to the fitted values exactly. Component sds rank "
            "them but do NOT partition variance: trend and Fourier "
            "terms both absorb slow drift.")


# compact alias per ledger/NAMING.md
additivecomponents = additive_components

# public names resolved by fn/_lazy_map.json
facebook_prophet = additive_components
