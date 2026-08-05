# morie.fn -- function file (rootcoder007/morie)
"""Control function approach to endogeneity."""

from . import _array_core as np
from . import _horowitz as HZ
from . import _hrz3 as H

from ._richresult import RichResult

__all__ = ["horowitz_control_function"]


def horowitz_control_function(x, y, w, bandwidth=None, iters=30):
    r"""Nonparametric control-function estimator.

    Horowitz (2009), Section 5.5.2, pages 186-187.  The model is

    .. math:: Y = g(X) + U,\quad
              E(U|V=v, W=w) = E(U|V=v),\quad V = X - E(X|W)   \quad (5.92)

    which is NON-NESTED with the NPIV model (5.4): it does not require
    :math:`E(U|W=w) = 0`, and that condition does not imply (5.92)
    either.  Conditioning on :math:`(X,W)` is equivalent to
    conditioning on :math:`(V,W)`, so with :math:`h(v) = E(U|V=v)`

    .. math:: E(Y|X, V) = g(X) + h(V)                         \quad (5.93)

    which is a nonparametric ADDITIVE model, estimated here by
    backfitting with local-linear smoothers (Chapter 3, as the text
    directs).  :math:`V` is not observed, so it is replaced by
    :math:`\hat V = X - \hat E(X|W)` with a nonparametric first stage,
    following Newey, Powell and Vella (1999).

    ``h`` is called a control function because it "controls" for the
    influence of :math:`X` on :math:`U`; it is not a nuisance term to
    be discarded, and it is returned.

    This is NOT the parametric Rivers-Vuong control function (see
    ``ctrfn``), in which both stages are linear and the second stage is
    a single added regressor.  Here both stages are nonparametric and
    nothing is assumed linear; the two coincide only when :math:`g` and
    :math:`E(X|W)` happen to be affine.

    Additive models identify :math:`g` and :math:`h` only up to a
    constant that can be shifted between them, so both are returned
    mean-zero with the level carried by ``intercept``.

    Parameters
    ----------
    x : array-like, shape (n,)
        Endogenous regressor.
    y : array-like, shape (n,)
        Response.
    w : array-like, shape (n,)
        Instrument.
    bandwidth : float, optional
        Common bandwidth.  Default: Silverman's rule per variable.
    iters : int, default 30
        Backfitting sweeps.  A FIXED count with no tolerance-based
        exit, so both language arms take the same path.

    Returns
    -------
    RichResult
        keys: ``g_hat``, ``h_hat``, ``v_hat``, ``intercept``,
        ``fitted``, ``resid_sd``, ``bandwidth_x``, ``bandwidth_v``,
        ``bandwidth_w``, ``n``, ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 5.5.2, eqs. (5.92)-(5.93),
    pp. 186-187.
    Newey, W. K., Powell, J. L. & Vella, F. (1999). Nonparametric
    estimation of triangular simultaneous equations models.
    *Econometrica* 67(3), 565-603.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = int(x.size)
    if y.size != n or w.size != n:
        raise ValueError(
            f"x, y, w must have the same length; got {n}, {y.size}, {w.size}.")
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    iters = int(iters)
    if iters < 1:
        raise ValueError(f"iters must be at least 1, got {iters}.")

    hw = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(w)
    hx = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(x)

    # First stage: V_hat = X - E_hat(X | W).
    ex_w = H.ll_smooth(w, x, w, hw)
    v_hat = np.asarray([float(x[i]) - float(ex_w[i]) for i in range(n)],
                       dtype=float)

    hv = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(v_hat)

    # Second stage: backfit the additive model (5.93).
    ybar = 0.0
    for t in y:
        ybar += float(t)
    ybar /= n
    g = [0.0] * n
    hh = [0.0] * n
    for _ in range(iters):
        rg = [float(y[i]) - ybar - hh[i] for i in range(n)]
        g = [float(t) for t in H.ll_smooth(x, rg, x, hx)]
        gm = sum(g) / n
        g = [t - gm for t in g]
        rh = [float(y[i]) - ybar - g[i] for i in range(n)]
        hh = [float(t) for t in H.ll_smooth(v_hat, rh, v_hat, hv)]
        hm = sum(hh) / n
        hh = [t - hm for t in hh]

    fitted = [ybar + g[i] + hh[i] for i in range(n)]
    ss = 0.0
    for i in range(n):
        ss += (float(y[i]) - fitted[i]) ** 2
    resid_sd = (ss / n) ** 0.5

    return RichResult(payload={
        "g_hat": g,
        "h_hat": hh,
        "v_hat": [float(t) for t in v_hat],
        "intercept": ybar,
        "fitted": fitted,
        "resid_sd": resid_sd,
        "bandwidth_x": hx,
        "bandwidth_v": hv,
        "bandwidth_w": hw,
        "n": n,
        "method": "Horowitz (2009) eqs. (5.92)-(5.93), nonparametric control function",
    })


def cheatsheet():
    return "hrzctrl: (5.93) E(Y|X,V) = g(X) + h(V); nonparametric, not Rivers-Vuong"
