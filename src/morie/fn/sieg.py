# morie.fn -- slice s05 (rootcoder007/morie)
"""Siegel repeated-median line: the 50%-breakdown robust regression.

Siegel, A. F. (1982), "Robust regression using repeated medians",
*Biometrika* 69(1), 242-244, doi:10.1093/biomet/69.1.242.

CITATION LIMIT, stated rather than papered over.  The Biometrika
article is closed access (Semantic Scholar and OpenAlex both report no
open-access location) and the Princeton technical-report version,
DTIC ADA092660, returns 403 to every fetch tried.  The estimator is
therefore taken from a source that states it in full and cites Siegel
for it: Borowski, M. and Fried, R. (2011), "Robust repeated median
regression in moving windows with data-adaptive width selection",
SFB 823 Discussion Paper 28/2011, TU Dortmund University,
doi:10.17877/de290r-13059, Section 2, equation (3), page 3, read off a
rendered page image.  That equation is

    beta_hat  = med_i  med_{i' != i}  (y_i - y_i') / (x_i - x_i')
    level_hat = med_i  ( y_i - beta_hat * x_i )

and the same page states the property that makes the estimator worth
having: a finite-sample replacement breakdown point of floor(n/2)/n,
about 50%, which Davies and Gather (2005) show is the maximum possible
for a regression-equivariant estimator.

The doubled median is the whole idea.  Theil-Sen takes ONE median over
all n(n-1)/2 pairwise slopes and breaks down at about 29%; taking a
median within each point first, and then across points, means a
minority of arbitrarily corrupted observations cannot move either
level of the calculation.  Half the sample can be replaced by
nonsense and the line does not move to infinity.

Points sharing an x value with no other point contribute no inner
median and are skipped; if no pair of distinct x values exists at all,
no slope is defined and that is an error rather than a number.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["siegel_repeated"]


def siegel_repeated(x, y):
    """Repeated-median slope and level for a simple linear fit.

    Parameters
    ----------
    x, y : array-like
        Predictor and response, same length, at least two points with
        two distinct x values.

    Returns
    -------
    RichResult
        keys: ``estimate`` (the slope), ``slope``, ``intercept``,
        ``fitted``, ``residuals``, ``n``, ``n_used``,
        ``breakdown_point``, ``method``.

    References
    ----------
    Siegel, A. F. (1982), *Biometrika* 69(1):242-244,
    doi:10.1093/biomet/69.1.242; the form used is equation (3) of
    Borowski and Fried (2011), doi:10.17877/de290r-13059.
    """
    xv = core.vec(x)
    yv = core.vec(y)
    if len(xv) != len(yv):
        raise ValueError("siegel_repeated: x and y must have the same length")
    n = len(xv)
    if n < 2:
        raise ValueError("siegel_repeated: need at least two points")
    inner = []
    for i in range(n):
        s = [(yv[j] - yv[i]) / (xv[j] - xv[i])
             for j in range(n) if xv[j] != xv[i]]
        if s:
            inner.append(core.median(s))
    if not inner:
        raise ValueError(
            "siegel_repeated: every x is identical, so no pairwise slope "
            "exists and no line is defined")
    slope = core.median(inner)
    intercept = core.median([yv[i] - slope * xv[i] for i in range(n)])
    fitted = [intercept + slope * v for v in xv]
    resid = [yv[i] - fitted[i] for i in range(n)]
    return RichResult(payload={
        "estimate": slope, "slope": slope, "intercept": intercept,
        "fitted": fitted, "residuals": resid,
        "n": int(n), "n_used": int(len(inner)),
        "breakdown_point": (n // 2) / n,
        "method": "Siegel (1982) repeated-median line, breakdown floor(n/2)/n"})


def cheatsheet():
    return ("sieg: two nested medians, not one -- half the sample can be "
            "nonsense and the line still holds; Theil-Sen breaks at 29%")


# compact alias per ledger/NAMING.md
siegelrepeated = siegel_repeated
