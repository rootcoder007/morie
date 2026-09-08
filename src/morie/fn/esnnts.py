# morie.fn -- function file (rootcoder007/morie)
"""Echo state network (reservoir computing)."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["echo_state_network"]

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _draw(k):
    """Deterministic weight in (-1, 1).

    Successive weights are taken from DIFFERENT van der Corput bases so
    that neighbouring reservoir entries are not correlated the way a
    single low-discrepancy stream would make them.
    """
    b = _PRIMES[k % len(_PRIMES)]
    return 2.0 * core.vdc(k // len(_PRIMES) + 1, b) - 1.0


def echo_state_network(y, reservoir_size=20, spectral_radius=0.9, leak=1.0,
                       ridge=1e-6, washout=None):
    """
    Echo state network (reservoir computing)

    Formula: a fixed random recurrent reservoir with a trained linear
    readout,

        x_t = (1 - a) x_{t-1} + a tanh(W x_{t-1} + W_in u_t)
        yhat_t = v' [1; x_t]

    driven in one-step prediction mode (u_t = y_{t-1}, target y_t).  Only
    v is fitted, by ridge regression, which is the whole point of the
    method.  W and W_in are fixed once and never trained.

    The reservoir is generated deterministically -- entry k is drawn from
    van der Corput base ``_PRIMES[k mod 12]`` -- so both language arms
    build the identical network.  W is then rescaled so that its induced
    infinity norm equals ``spectral_radius``; ||W||_inf < 1 is Jaeger's
    sufficient condition for the echo state property (2001, Prop. 3), a
    stronger and cheaper guarantee than the spectral-radius rule of thumb.

    Parameters
    ----------
    y : array-like
        Series (at least 3 points).
    reservoir_size : int
        Number of reservoir units.
    spectral_radius : float
        Target induced infinity norm of W (>= 0).
    leak : float
        Leaking rate a in (0, 1].
    ridge : float
        Tikhonov regularisation for the readout (>= 0).
    washout : int, optional
        Initial steps discarded before fitting.  Default
        min(reservoir_size, n // 4).

    Returns
    -------
    result : dict
        Keys: estimate (in-sample MSE), mse, nrmse, coef, win, size,
        washout, nfit, n, method.

    References
    ----------
    Jaeger (2001), GMD Report 148, German National Research Center for
    Information Technology.
    Jaeger & Haas (2004), Science 304(5667):78-80,
    doi:10.1126/science.1091277.
    """
    y = [float(v) for v in y]
    n = len(y)
    if n < 3:
        raise ValueError("need at least 3 observations")
    size = int(reservoir_size)
    if size < 1:
        raise ValueError("reservoir_size must be positive")
    sr = float(spectral_radius)
    if sr < 0.0:
        raise ValueError("spectral_radius must be non-negative")
    a = float(leak)
    if not (0.0 < a <= 1.0):
        raise ValueError("leak must lie in (0, 1]")
    lam = float(ridge)
    if lam < 0.0:
        raise ValueError("ridge must be non-negative")
    wo = (min(size, n // 4) if washout is None else int(washout))
    if wo < 0 or wo >= n - 1:
        raise ValueError("washout must lie in [0, n-1)")
    W = [[_draw(i * size + j) for j in range(size)] for i in range(size)]
    Win = [_draw(size * size + i) for i in range(size)]
    norm = max(sum(abs(v) for v in row) for row in W)
    scale = (sr / norm) if norm > 0.0 else 0.0
    W = [[v * scale for v in row] for row in W]
    x = [0.0] * size
    rows = []
    targ = []
    for t in range(1, n):
        u = y[t - 1]
        nx = []
        for i in range(size):
            z = Win[i] * u
            for j in range(size):
                z += W[i][j] * x[j]
            nx.append((1.0 - a) * x[i] + a * math.tanh(z))
        x = nx
        if t - 1 >= wo:
            rows.append([1.0] + list(x))
            targ.append(y[t])
    nfit = len(rows)
    k = size + 1
    if nfit < 1:
        raise ValueError("no rows left after washout")
    XtX = [[sum(rows[r][i] * rows[r][j] for r in range(nfit)) for j in range(k)]
           for i in range(k)]
    for i in range(k):
        XtX[i][i] += lam
    Xty = [sum(rows[r][i] * targ[r] for r in range(nfit)) for i in range(k)]
    v = core.cholsolve(XtX, Xty)
    fit = [sum(rows[r][i] * v[i] for i in range(k)) for r in range(nfit)]
    mse = sum((targ[r] - fit[r]) ** 2 for r in range(nfit)) / nfit
    mt = sum(targ) / nfit
    vt = sum((t - mt) ** 2 for t in targ) / nfit
    nrmse = math.sqrt(mse / vt) if vt > 0.0 else float("nan")
    return RichResult(payload={
        "estimate": mse,
        "mse": mse,
        "nrmse": nrmse,
        "coef": v,
        "win": Win,
        "size": size,
        "washout": wo,
        "nfit": nfit,
        "n": n,
        "method": "Echo state network (reservoir computing)",
    })


def cheatsheet():
    return "esnnts: Echo state network (reservoir computing)"


# compact alias per ledger/NAMING.md
echostatenetwork = echo_state_network
