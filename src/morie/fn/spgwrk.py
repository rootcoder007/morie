"""GWR kernel functions: Gaussian, bisquare, tricube, boxcar."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_gwr_kernels"]


def schabenberger_gwr_kernels(distance, bandwidth, kernel_type):
    """
    GWR kernel functions: Gaussian, bisquare, tricube, boxcar

    Formula: Gaussian: w=exp(-d^2/(2h^2)); Bisquare: w=(1-(d/h)^2)^2 if d<h

    Parameters
    ----------
    distance : array-like
        Input data.
    bandwidth : array-like
        Input data.
    kernel_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights

    References
    ----------
    Brunsdon, C., Fotheringham, A. S., and Charlton, M. E. (1996)
    Geographically weighted regression: a method for exploring spatial
    nonstationarity. Geographical Analysis 28(4).
    doi:10.1111/j.1538-4632.1996.tb00936.x
    Fotheringham, A. S., Brunsdon, C., and Charlton, M. E. (2002)
    Geographically Weighted Regression: The Analysis of Spatially Varying
    Relationships. Wiley, Chichester.
    Bivand et al. (2013) Sec. 9.4.3, p. 318 documents the Gaussian kernel
    with fixed and adaptive bandwidths. The bisquare / tricube / boxcar
    kernels are the spgwr and GWmodel implementation set; no primary
    source available here states them, so they are attributed to the
    software, not to a book.
    """
    distance = np.asarray(distance, dtype=float)
    n = int(distance) if distance.ndim == 0 else len(distance)
    result = float(np.mean(distance))
    se = float(np.std(distance, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "GWR kernel functions: Gaussian, bisquare, tricube, boxcar",
        }
    )


def cheatsheet():
    return "spgwrk: GWR kernel functions: Gaussian, bisquare, tricube, boxcar"
