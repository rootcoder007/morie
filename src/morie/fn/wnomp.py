# morie.fn -- function file (rootcoder007/morie)
"""W-NOMINATE vote probability from the Gaussian spatial utility."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wnominate_probability"]


def wnominate_probability(ideal_point, yea_pos, nay_pos, beta=15.0):
    r"""Choice probability under NOMINATE's Gaussian deterministic utility.

    .. math:: P_i(\text{Yea}) = \frac{e^{\beta u_{iy}}}
              {e^{\beta u_{iy}} + e^{\beta u_{in}}},
              \qquad u_{i\cdot} = e^{-\|x_i - z_\cdot\|^2/2},

    i.e. a logit over the Gaussian (not quadratic) utilities of the
    Yea and Nay outcome locations -- the distinguishing feature of the
    NOMINATE family versus quadratic-utility IRT models. Larger beta
    means less noise; as beta grows the choice becomes the
    deterministic nearer-outcome rule.

    Parameters
    ----------
    ideal_point : array-like, shape (k,) or (n, k)
        Legislator ideal point(s).
    yea_pos, nay_pos : array-like, shape (k,)
        Outcome coordinates of the Yea and Nay alternatives.
    beta : float, default 15.0
        Signal-to-noise weight (wnominate's default).

    Returns
    -------
    RichResult
        keys: ``p_yea`` (scalar or (n,)), ``u_yea``, ``u_nay``,
        ``beta``, ``method``.

    References
    ----------
    Poole, K. T. & Rosenthal, H. (1997). *Congress: A
    Political-Economic History of Roll Call Voting*. Oxford
    University Press. (the NOMINATE Gaussian utility)

    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Sec. 5.3 (W-NOMINATE), pp.
    139-144.
    """
    X = np.atleast_2d(np.asarray(ideal_point, dtype=float))
    zy = np.asarray(yea_pos, dtype=float).ravel()
    zn = np.asarray(nay_pos, dtype=float).ravel()
    if zy.shape != zn.shape or zy.size != X.shape[1]:
        raise ValueError("yea_pos and nay_pos must share the ideal point's dimension.")
    beta = float(beta)
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}.")

    uy = np.exp(-((X - zy) ** 2).sum(axis=1) / 2.0)
    un = np.exp(-((X - zn) ** 2).sum(axis=1) / 2.0)
    # logit over beta-scaled Gaussian utilities, stabilised
    a = beta * uy
    b = beta * un
    m = np.maximum(a, b)
    p = np.exp(a - m) / (np.exp(a - m) + np.exp(b - m))

    scalar = np.ndim(ideal_point) <= 1
    return RichResult(
        payload={
            "p_yea": float(p[0]) if scalar else p,
            "u_yea": float(uy[0]) if scalar else uy,
            "u_nay": float(un[0]) if scalar else un,
            "beta": beta,
            "method": "W-NOMINATE Gaussian-utility vote probability",
        }
    )


def cheatsheet():
    return "wnomp: P(yea) = logit over beta * exp(-||x-z||^2/2) utilities"
