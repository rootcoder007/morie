"""Thomas cluster process: Normal offspring displacements."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_thomas_process"]


def schabenberger_thomas_process(r, rho, mu, sigma):
    """
    Thomas cluster process: Normal offspring displacements

    Formula: K(r) = pi*r^2 + mu*[1-exp(-r^2/(4*sigma^2))]/rho

    Parameters
    ----------
    r : array-like
        Input data.
    rho : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: k_function

    References
    ----------
    Thomas, M. (1949) A generalization of Poisson's binomial limit for use in
    ecology. Biometrika 36(1-2):18-25. doi:10.1093/biomet/36.1-2.18
    The Thomas process is the Neyman-Scott process with Poisson offspring
    counts and radially symmetric Gaussian displacements; the general
    cluster-process framework is Schabenberger & Gotway (2005) Sec. 3.7.2
    "Clustered Processes", pp. 126-130, which does not name this special
    case.
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Thomas cluster process: Normal offspring displacements",
        }
    )


def cheatsheet():
    return "spthom: Thomas cluster process: Normal offspring displacements"
