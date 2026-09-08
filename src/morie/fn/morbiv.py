"""Bivariate Moran I between two variables (Anselin-Syabri-Smirnov)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["morbiv", "bivariate_morans_i"]


def morbiv(x, y, W, scale=True, row_standardize=True):
    """
    Bivariate Moran I: spatial correlation between x and the lag of y.

        I_B = sum_i x_i (W y)_i / sum_i x_i^2

    "the slope of a regression of Wy on x", with both variables
    standardized and W row-standardized. Note the caveat of the source:
    I_B ignores the in-place correlation between x_i and y_i, and can
    overstate spatial association when that is strong.

    Standardization follows the spdep::moran_bv reference
    implementation: R scale(), i.e. mean 0 and UNIT SAMPLE variance
    (divisor n-1). GeoDa states "means zero and variance one" without
    fixing the divisor; the n-1 choice is adopted here for anchoring.

    Sources
    -------
    Anselin, L., Syabri, I. & Smirnov, O. (2002). Visualizing
    multivariate spatial correlation with dynamically linked windows.
    In *New Tools for Spatial Data Analysis*, CSISS, Santa Barbara.
    Anselin, L. GeoDa workbook, "Global Spatial Autocorrelation (2)",
    bivariate Moran scatter plot (fetched-wave3/
    anselin-geoda-workbook-lab5b-bivariate-morans-i.html).
    Reference implementation: spdep::moran_bv (CRAN, source read
    directly; statistic sum(x * lag(y)) / sum(x^2) after scale()).

    Parameters
    ----------
    x : array-like, (n,)
        Focal variable.
    y : array-like, (n,)
        Lagged variable.
    W : array-like, (n, n)
        Spatial weights; row-standardized internally by default.
    scale : bool
        Standardize x and y (mean 0, sd with divisor n-1).
    row_standardize : bool
        Divide each row of W by its sum (zero rows left as zero).

    Returns
    -------
    RichResult
        Keys: statistic, lag_y, x_std, y_std.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    W = np.asarray(W, dtype=float)
    n = x.size
    if y.size != n:
        raise ValueError("`x` and `y` must have equal length")
    if W.shape != (n, n):
        raise ValueError(f"W must be ({n}, {n}), got {W.shape}")
    if row_standardize:
        rs = W.sum(axis=1)
        rs_safe = np.where(rs == 0, 1.0, rs)
        W = W / rs_safe[:, None]
    if scale:
        x = (x - float(np.mean(x))) / float(np.std(x, ddof=1))
        y = (y - float(np.mean(y))) / float(np.std(y, ddof=1))
    lag_y = W @ y
    stat = float(np.sum(x * lag_y) / np.sum(x**2))
    return RichResult(payload={
        "statistic": stat, "lag_y": lag_y, "x_std": x, "y_std": y, "n": int(n),
        "method": "Bivariate Moran I (Anselin-Syabri-Smirnov 2002)",
    })


# long descriptive alias (stub-era name)
bivariate_morans_i = morbiv


def cheatsheet():
    return "morbiv: bivariate Moran I = sum x (Wy) / sum x^2, standardized"
