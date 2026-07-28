# morie.fn -- function file (rootcoder007/morie)
"""Kendall's tau_b with tie correction."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_kendall_ties"]


def gibbons_kendall_ties(x, y):
    r"""Tie-corrected Kendall coefficient tau_b.

    .. math:: \tau_b = \frac{P - Q}
              {\sqrt{(P + Q + T_x)(P + Q + T_y)}},

    where :math:`T_x` counts pairs tied on x only and :math:`T_y`
    pairs tied on y only (Gibbons Ch. 11.2, treatment of ties). Pairs
    tied on both variables drop out of every term. Without ties
    tau_b equals the plain tau, which the tests assert.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Paired observations.

    Returns
    -------
    RichResult
        keys: ``tau_b``, ``P``, ``Q``, ``T_x``, ``T_y``, ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.2.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if y.size != n:
        raise ValueError("x and y must have the same length.")
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}.")

    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(n, 1)
    dxu, dyu = dx[iu], dy[iu]
    P = int(np.sum((dxu * dyu) > 0))
    Q = int(np.sum((dxu * dyu) < 0))
    Tx = int(np.sum((dxu == 0) & (dyu != 0)))
    Ty = int(np.sum((dyu == 0) & (dxu != 0)))
    denom = np.sqrt(float(P + Q + Tx) * float(P + Q + Ty))
    if denom == 0:
        raise ValueError("all pairs are tied; tau_b is undefined.")
    return RichResult(
        payload={
            "tau_b": float((P - Q) / denom), "P": P, "Q": Q,
            "T_x": Tx, "T_y": Ty, "n": int(n),
            "method": "Kendall tau_b = (P-Q)/sqrt((P+Q+Tx)(P+Q+Ty))",
        }
    )


def cheatsheet():
    return "gb1122t: tau_b; both-tied pairs drop from every term"
