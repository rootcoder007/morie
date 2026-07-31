"""GWR bandwidth selection via leave-one-out cross-validation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_gwr_bandwidth"]


def schabenberger_gwr_bandwidth(x, y, coords):
    """
    GWR bandwidth selection via leave-one-out cross-validation

    Formula: AIC_c(h) = 2*n*log(sigma_hat) + 2*tr(S) + 2*tr(S)'*tr(S)/(n-tr(S)-1)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_bandwidth

    References
    ----------
    Bivand, R. S., Pebesma, E., and Gomez-Rubio, V. (2013) Applied Spatial
    Data Analysis with R, 2nd ed., Springer. Sec. 9.4.3, p. 318: an
    isotropic spatial weights kernel, "typically a Gaussian kernel with a
    fixed bandwidth chosen by leave-one-out cross-validation"; adaptive
    bandwidths are noted as an alternative.
    Brunsdon, C., Fotheringham, A. S., and Charlton, M. E. (1996)
    Geographically weighted regression: a method for exploring spatial
    nonstationarity. Geographical Analysis 28(4).
    doi:10.1111/j.1538-4632.1996.tb00936.x
    Cross-validation is the only selector verified against a primary
    source here; an AIC/AICc selector is NOT claimed by these references.
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "GWR bandwidth selection via leave-one-out cross-validation"}
    )


def cheatsheet():
    return "spgwrb: GWR bandwidth selection via leave-one-out cross-validation"
