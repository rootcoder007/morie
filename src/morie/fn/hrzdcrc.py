# morie.fn -- function file (rootcoder007/morie)
"""Deconvolution convergence rates."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_deconv_rate"]


def hrz_deconv_rate(n, error="normal", s=2.0, r=2.0):
    r"""Convergence rates for deconvolution (Horowitz Ch. 5):

    .. math:: \|\hat f_U - f_U\|^2 =
              \begin{cases}
              O_p(n^{-r}) & \text{ordinary smooth}\\
              O_p\big[(\log n)^{-s}\big] & \text{supersmooth}
              \end{cases}

    The gap is enormous and is the practical message of the chapter:
    at n = 10^6 a logarithmic rate has barely moved. Both are returned
    at the requested n so the difference is a number rather than a
    footnote.

    Parameters
    ----------
    n : int
        Sample size.
    error : {"normal", "laplace"}
        Error type.
    s : float, default 2.0
        Supersmooth exponent.
    r : float, default 2.0
        Ordinary-smooth exponent.

    Returns
    -------
    RichResult
        keys: ``rate``, ``regime``, ``polynomial_rate``,
        ``logarithmic_rate``, ``ratio``, ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 5 (rates of convergence in deconvolution).
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    if error not in ("normal", "laplace"):
        raise ValueError("error must be 'normal' or 'laplace'.")
    poly = float(n ** (-float(r)))
    logr = float(np.log(n) ** (-float(s)))
    supersmooth = error == "normal"
    return RichResult(payload={"rate": logr if supersmooth else poly,
                               "regime": "supersmooth" if supersmooth
                               else "ordinary smooth",
                               "polynomial_rate": poly, "logarithmic_rate": logr,
                               "ratio": logr / poly if poly > 0 else np.inf,
                               "n": n,
                               "method": "n^{-r} vs (log n)^{-s}; the gap is the chapter's point"})


def cheatsheet():
    return "hrzdcrc: at n=1e6 the logarithmic rate has barely moved"
