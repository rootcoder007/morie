"""Periodic covariance kernel (MacKay 1998; Rasmussen & Williams 2006)."""

import math

from ._richresult import RichResult

__all__ = ["perK", "periodic_kernel"]


def perK(x1, x2=None, period=1.0, lengthscale=1.0, variance=1.0):
    """
    Periodic kernel matrix between two sets of scalar inputs.

    MacKay's (1998) warping construction, as given in Rasmussen &
    Williams (2006, Sec. 4.2.3): map the input x to
    u(x) = (cos(2 pi x / p), sin(2 pi x / p)) and apply the squared
    exponential in u-space, which yields

        k(x, x') = sigma^2 exp( - 2 sin^2( pi (x - x') / p ) / l^2 ).

    The kernel is positive definite, has period p in x - x'
    (k(x + p, x') = k(x, x')), attains its maximum sigma^2 exactly at
    x - x' in p Z, and reduces toward the squared-exponential kernel
    locally for |x - x'| << p.

    Sources
    -------
    MacKay, D. J. C. (1998). Introduction to Gaussian processes. In
    C. M. Bishop (ed.), *Neural Networks and Machine Learning*,
    NATO ASI Series, Springer (the (cos, sin) warping; local copy
    fetched-wave3/mackay-1998-gp-intro.pdf).
    Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian
    Processes for Machine Learning*, MIT Press, Sec. 4.2.3 (the
    explicit periodic form; local copy
    fetched-wave3/rasmussen-williams-2006-gpml-ch4.pdf).

    Parameters
    ----------
    x1 : sequence of float
        First input set.
    x2 : sequence of float, optional
        Second input set (defaults to x1, giving the Gram matrix).
    period : float
        Period p > 0.
    lengthscale : float
        Lengthscale l > 0 in the warped space.
    variance : float
        Signal variance sigma^2 > 0.

    Returns
    -------
    RichResult
        Keys: K (matrix as list of rows), shape, diag_is_variance.
    """
    a = [float(v) for v in x1]
    b = a if x2 is None else [float(v) for v in x2]
    p = float(period)
    l = float(lengthscale)
    s2 = float(variance)
    if p <= 0 or l <= 0 or s2 <= 0:
        raise ValueError("period, lengthscale, variance must be positive")
    K = []
    for xa in a:
        row = []
        for xb in b:
            s = math.sin(math.pi * (xa - xb) / p)
            row.append(s2 * math.exp(-2.0 * s * s / (l * l)))
        K.append(row)
    diag_ok = x2 is None and all(
        abs(K[i][i] - s2) < 1e-15 for i in range(len(a)))
    return RichResult(payload={
        "K": K,
        "shape": (len(a), len(b)),
        "period": p, "lengthscale": l, "variance": s2,
        "diag_is_variance": diag_ok,
        "method": "periodic kernel (MacKay 1998; R&W 2006 Sec. 4.2.3)",
    })


# long descriptive alias (stub-era name)
periodic_kernel = perK


def cheatsheet():
    return "perK: k = s2 exp(-2 sin^2(pi (x-x')/p) / l^2)"
