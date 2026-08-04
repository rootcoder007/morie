# morie.fn -- function file (rootcoder007/morie)
"""Radial basis (Gaussian) kernel between two vectors.

Standard kernel method.  Triage confirmed this names no owning
source; the Gaussian kernel predates any single reference that could
be attached to it, so none is manufactured.

Parameterization warning.  Two conventions are in circulation:
exp(-||x-y||^2 / (2 sigma^2)) with a bandwidth, and
exp(-gamma ||x-y||^2) with a rate.  This function takes the
bandwidth, which is what its signature declares; the equivalent
gamma = 1/(2 sigma^2) is returned so the two cannot be confused.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rbf_kernel"]


def rbf_kernel(x, y, sigma):
    """Gaussian kernel k(x, y) = exp(-||x - y||^2 / (2 sigma^2)).

    The value is 1 when the two vectors coincide and decays to 0 as
    they separate, so it acts as a similarity: sigma sets how far
    apart two points can be before they stop counting as neighbours.

    Parameters
    ----------
    x, y : array-like vectors of the same length.
    sigma : float, the bandwidth; must be positive.

    Returns
    -------
    RichResult with keys estimate (the kernel value), value,
    sq_distance, distance, sigma, gamma, method.
    """
    a = [float(v) for v in x]
    b = [float(v) for v in y]
    if len(a) != len(b):
        raise ValueError("x and y must have the same length")
    s = float(sigma)
    if s <= 0:
        raise ValueError("sigma must be positive")
    d2 = sum((a[i] - b[i]) ** 2 for i in range(len(a)))
    g = 1.0 / (2.0 * s * s)
    k = math.exp(-d2 * g)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(k), "value": float(k),
        "sq_distance": float(d2), "distance": math.sqrt(d2),
        "sigma": s, "gamma": float(g),
        "method": "radial basis (Gaussian) kernel",
    }), "rbfk")


def cheatsheet():
    return "rbfk: RBF (Gaussian) kernel"


# compact alias per ledger/NAMING.md
rbfkern = rbf_kernel
