# morie.fn -- function file (rootcoder007/morie)
"""Transfer entropy (information-theoretic causality)."""

import numpy as np

from ._richresult import RichResult
from .granci import granger_causality_info

__all__ = ["transfer_entropy"]


def transfer_entropy(x, y, lag=1, method="gaussian", bins=4):
    r"""Transfer entropy from x to y.

    Schreiber's measure

    .. math:: T_{X \to Y} = H(Y_t \mid Y_{t-1..t-k})
              - H(Y_t \mid Y_{t-1..t-k}, X_{t-1..t-k})

    quantifies the reduction in uncertainty about :math:`Y_t` from
    knowing X's past over and above Y's own past.

    ``method="gaussian"`` uses the closed form
    :math:`\tfrac12 \ln(\mathrm{RSS}_r/\mathrm{RSS}_u)` -- exactly the
    Gaussian CMI, which Barnett-Barrett-Seth prove equals transfer
    entropy for Gaussian processes. ``method="binned"`` is the
    nonparametric plug-in on quantile-binned data (lag 1 histories),
    which also sees nonlinear coupling a linear model misses.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Source and target series.
    lag : int, default 1
        History length k (binned mode uses k = 1).
    method : {"gaussian", "binned"}
    bins : int, default 4
        Quantile bins per variable in binned mode.

    Returns
    -------
    RichResult
        keys: ``te`` (nats), ``p_value`` (gaussian mode; None for
        binned), ``n``, ``lag``, ``bins`` (binned mode; else None),
        ``method``.

    References
    ----------
    Schreiber, T. (2000). Measuring information transfer. *Physical
    Review Letters*, 85(2), 461-464. doi:10.1103/PhysRevLett.85.461.

    Barnett, L., Barrett, A. B. & Seth, A. K. (2009). Granger
    causality and transfer entropy are equivalent for Gaussian
    variables. *Physical Review Letters*, 103(23), 238701.
    """
    if method == "gaussian":
        g = granger_causality_info(x, y, lag=lag)
        return RichResult(
            payload={
                "te": g["mi"],
                "p_value": g["p_value"],
                "n": g["n"],
                "lag": g["lag"],
                "bins": None,
                "method": "Transfer entropy (Gaussian closed form == Granger CMI)",
            }
        )
    if method != "binned":
        raise ValueError(f"method must be 'gaussian' or 'binned', got {method!r}.")

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have equal length.")
    b = int(bins)
    if b < 2:
        raise ValueError(f"bins must be at least 2, got {b}.")
    n = y.size
    if n < 10 * b:
        raise ValueError(f"need at least {10 * b} observations for {b} bins, got {n}.")

    def discretise(v):
        edges = np.quantile(v, np.linspace(0, 1, b + 1)[1:-1])
        return np.searchsorted(edges, v)

    xd, yd = discretise(x), discretise(y)
    yt, yp, xp = yd[1:], yd[:-1], xd[:-1]

    def H(*cols):
        keys = np.stack(cols, axis=1)
        _, counts = np.unique(keys, axis=0, return_counts=True)
        pr = counts / counts.sum()
        return float(-(pr * np.log(pr)).sum())

    # T = H(Yt, Yp) - H(Yp) - H(Yt, Yp, Xp) + H(Yp, Xp)
    te = H(yt, yp) - H(yp) - H(yt, yp, xp) + H(yp, xp)

    return RichResult(
        payload={
            "te": float(max(te, 0.0)),
            "p_value": None,
            "n": int(n),
            "lag": 1,
            "bins": b,
            "method": "Transfer entropy (quantile-binned plug-in, k=1)",
        }
    )


def cheatsheet():
    return "trnfen: T_{X->Y}; gaussian closed form or binned plug-in (Schreiber 2000)"
