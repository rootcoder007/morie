# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Life satisfaction = theta0 + theta1 * GDP_per_capita (introductory example)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_linear_regression_life"]

_METHOD = "Univariate linear model (life satisfaction vs GDP per capita)"


def geron_linear_regression_life(gdp, theta0, theta1, life_sat=None):
    """
    Life satisfaction = theta0 + theta1 * GDP_per_capita (introductory example).

    Formula: life_sat = theta0 + theta1 * gdp

    The book's opening example.  Given the two parameters this predicts
    life satisfaction for each GDP figure; if the observed satisfaction
    is supplied as well, the residuals, RMSE and R^2 come back too, so
    the "is this a good model?" question has an answer rather than a
    picture.  ``theta1`` is the satisfaction points gained per unit of
    GDP per capita -- units matter, and a tiny slope usually just means
    GDP was passed in dollars rather than thousands.

    Parameters
    ----------
    gdp : array-like
        GDP per capita, one entry per country.
    theta0 : float
        Intercept.
    theta1 : float
        Slope.
    life_sat : array-like, optional
        Observed life satisfaction, for residual diagnostics.

    Returns
    -------
    result : RichResult
        Keys: prediction, residuals, rmse, r2, theta0, theta1,
        estimate, n, method.

    Examples
    --------
    ``4.85 + 4.91e-5 * 40000 = 6.814``:

    >>> r = geron_linear_regression_life([40000.0], theta0=4.85, theta1=4.91e-5)
    >>> round(float(r["prediction"][0]), 6)
    6.814

    With observations supplied the residuals are exact:

    >>> r2 = geron_linear_regression_life([0.0, 1.0, 2.0], theta0=1.0, theta1=2.0,
    ...                                   life_sat=[1.0, 4.0, 5.0])
    >>> [float(v) for v in r2["prediction"]]
    [1.0, 3.0, 5.0]
    >>> [float(v) for v in r2["residuals"]]
    [0.0, 1.0, 0.0]
    >>> round(r2["rmse"], 6)
    0.57735

    References
    ----------
    Géron Ch 1
    """
    x = np.atleast_1d(np.asarray(gdp, dtype=float)).ravel()
    if x.size == 0:
        raise ValueError("geron_linear_regression_life: gdp is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("geron_linear_regression_life: gdp contains non-finite values")
    t0, t1 = float(theta0), float(theta1)
    if not np.isfinite(t0) or not np.isfinite(t1):
        raise ValueError("geron_linear_regression_life: theta0 and theta1 must be finite")

    pred = t0 + t1 * x
    resid = rmse = r2 = None
    if life_sat is not None:
        y = np.atleast_1d(np.asarray(life_sat, dtype=float)).ravel()
        if y.size != x.size:
            raise ValueError(
                f"geron_linear_regression_life: life_sat has {y.size} entries but gdp has {x.size}"
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("geron_linear_regression_life: life_sat contains non-finite values")
        resid = y - pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            raise ValueError(
                "geron_linear_regression_life: life_sat has zero variance, so R^2 is undefined"
            )
        r2 = 1.0 - float(np.sum(resid**2)) / ss_tot

    lines = [("theta0 (intercept)", t0), ("theta1 (slope)", t1), ("Countries", int(x.size))]
    if rmse is not None:
        lines += [("RMSE", rmse), ("R^2", r2)]

    return RichResult(
        title="Life satisfaction vs GDP",
        summary_lines=lines,
        interpretation=(
            "A model-based fit: two parameters chosen to minimise a cost, "
            "as against geron_instance_based which stores the data instead."
        ),
        payload={
            "prediction": pred,
            "residuals": resid,
            "rmse": rmse,
            "r2": r2,
            "theta0": t0,
            "theta1": t1,
            "estimate": float(pred[0]),
            "n": int(x.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlrl: life_sat = theta0 + theta1*gdp, with residuals/RMSE/R^2 when observations are given"
