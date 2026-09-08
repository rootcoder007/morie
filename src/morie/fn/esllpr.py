# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local linear regression (ESL Ch 6.1.1)."""

from . import _array_core as np

from ._richresult import RichResult
from .eslnnk import _kernel_weights, _KERNELS

__all__ = ["esl_local_linear"]


def esl_local_linear(x0, x, y, lambda_, kernel="epanechnikov"):
    """
    Local linear fit: minimise
    sum K_l(x0, xi) (yi - alpha - beta xi)^2, report alpha + beta x0.

    This exists to fix the boundary bias of the Nadaraya-Watson
    smoother (eslnnk). ESL Ch 6.1.1 shows local linear regression
    corrects that bias to FIRST ORDER automatically — the property is
    worth stating because it is not obvious and it is the entire
    reason to prefer this estimator: on data lying exactly on a line
    the local linear fit is exact EVERYWHERE, including at the ends,
    while the kernel average is not.

    The returned slope is a bonus: it estimates the derivative of the
    regression function at x0.

    Parameters
    ----------
    x0 : array-like
        Query point(s).
    x, y : array-like
        Training pairs.
    lambda_ : float
        Bandwidth, > 0.
    kernel : str
        As in eslnnk.

    Returns
    -------
    result : dict
        Keys: estimate (fit at first x0), values, slopes, n_in_window,
        lambda, kernel, n, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 6.1.1 (Eq. 6.8).

    Examples
    --------
    Exact on a line, including at the boundary where the kernel
    average fails:

    >>> xs = [0.0, 1.0, 2.0, 3.0]
    >>> ys = [1.0, 3.0, 5.0, 7.0]
    >>> round(esl_local_linear(0.0, xs, ys, 1.5)["estimate"], 10)
    1.0
    >>> round(esl_local_linear(0.0, xs, ys, 1.5)["slopes"][0], 10)
    2.0

    The Nadaraya-Watson estimate at the same boundary point is pulled
    toward the interior:

    >>> from morie.fn.eslnnk import esl_nadaraya_watson
    >>> esl_nadaraya_watson(0.0, xs, ys, 1.5)["estimate"] > 1.0
    True
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    xd = np.atleast_1d(np.asarray(x, dtype=float))
    yd = np.atleast_1d(np.asarray(y, dtype=float))
    lam = float(lambda_)
    if xd.size != yd.size:
        raise ValueError(f"x ({xd.size}) and y ({yd.size}) lengths differ.")
    if lam <= 0:
        raise ValueError(f"the bandwidth must be positive; got {lam}.")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}; got '{kernel}'.")
    vals, slopes, cnt = [], [], []
    for q in x0:
        w = _kernel_weights(np.abs(q - xd) / lam, kernel)
        cnt.append(int(np.sum(w > 0)))
        if np.sum(w > 0) < 2:
            vals.append(float("nan")); slopes.append(float("nan")); continue
        D = np.column_stack([np.ones_like(xd), xd - q])
        sw = np.sqrt(w)
        qr = np.linalg.lstsq(D * sw[:, None], yd * sw, rcond=None)
        b = qr[0]
        vals.append(float(b[0]))            # centred at q, so alpha IS the fit
        slopes.append(float(b[1]))
    return RichResult(payload={
        "estimate": vals[0], "values": vals, "slopes": slopes,
        "n_in_window": cnt, "lambda": lam, "kernel": kernel, "n": int(xd.size),
        "method": "local linear WLS on (x - x0); corrects NW boundary bias to first order"})


def cheatsheet():
    return "esllpr: WLS on centred x; exact on lines even at boundaries; slope = derivative"


# compact alias per ledger/NAMING.md
esllocallinear = esl_local_linear
