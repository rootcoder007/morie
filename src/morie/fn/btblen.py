# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
r"""Automatic block-length selection via flat-top lag-windows.

Politis, D. N. and White, H. (2004), "Automatic Block-Length Selection
for the Dependent Bootstrap", *Econometric Reviews* 23(1), 53-70.  Every
formula below was read from rendered images of the journal PDF, pages
58, 59 and 60 (article pages), because the text layer of that PDF drops
minus signs.

The stub this replaces was labelled "Politis-Romano (2009)" with the
form ell = round(C n^(1/5)).  The automatic, spectral-density-based
selector the label points at is Politis and White (2004); the n^(1/5)
rate belongs to distribution-function estimation, whereas the selector
below is the MSE-optimal one for the variance, whose rate is n^(1/3).
The paper is cited, the rate is the paper's, and the discrepancy is
recorded here rather than papered over.

Page 58, the flat-top lag-window of Politis and Romano (1995):

.. code-block:: text

    lambda(t) = 1            if |t| in [0, 1/2]
              = 2 (1 - |t|)  if |t| in [1/2, 1]
              = 0            otherwise

with R_hat(k) = N^-1 sum_{i=1}^{N-|k|} (X_i - Xbar)(X_{i+|k|} - Xbar),
and equation (8),

    G_hat   = sum_{k=-M}^{M} lambda(k/M) |k| R_hat(k),
    g_hat(w)= sum_{k=-M}^{M} lambda(k/M) R_hat(k) cos(wk),
    D_hat_SB = 4 g_hat^2(0) + (2/pi) int_{-pi}^{pi} (1 + cos w) g_hat^2(w) dw,

equation (9),  b_opt_SB = (2 G_hat^2 / D_hat_SB)^(1/3) N^(1/3),
page 60 equation (13),  D_hat_CB = (4/3) g_hat^2(0),
and equation (14),  b_opt_CB = [ (2 G_hat^2 / D_hat_CB)^(1/3) N^(1/3) ],
where [x] is the nearest integer.  Page 62 notes the moving-block optimum
is the same as the circular one, so ``ell`` serves both.

The bandwidth M is chosen by the correlogram rule of the footnote to
section 3.2 (page 59): m_hat is the smallest positive integer with
|rho_hat(m_hat + k)| < c sqrt(log10(N) / N) for k = 1, ..., K_N, with the
paper's recommended c = 2 and K_N = max(5, sqrt(log10 N)), and M = 2 m_hat.

The integral is evaluated on a fixed 2000-panel trapezoid grid over
[-pi, pi] so that both language arms produce identical numbers; the
integrand is a smooth trigonometric polynomial of degree 2M + 1, for
which that grid is far finer than needed.

Anchor: for white noise R(k) = 0 for k != 0, so G_hat = 0 and both
optimal block lengths collapse to 1 -- the iid bootstrap, which is
correct for independent data.  For an AR(1) with parameter rho the
theoretical G = sum |k| R(k) = 2 sigma_e^2 rho / ((1-rho^2)(1-rho)^2) and
g(0) = sigma_e^2 / (1-rho)^2, both closed forms, and ``btblen`` is
checked against the block length they imply.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_block_length_pr"]

_NPANEL = 2000


def _lam(t):
    at = abs(t)
    if at <= 0.5:
        return 1.0
    if at <= 1.0:
        return 2.0 * (1.0 - at)
    return 0.0


def boot_block_length_pr(x, method="circular", c=2.0, m_max=None):
    """Politis-White automatic block length.

    Parameters
    ----------
    x : array-like
        The series, in time order.
    method : {"circular", "stationary", "moving"}
        Which optimum to report in ``ell``.  "moving" is the same as
        "circular" (Politis and White 2004, p.62).
    c : float
        Constant in the correlogram cutoff; the paper recommends 2.
    m_max : int, optional
        Cap on m_hat.  Defaults to ``ceil(sqrt(n)) + K_N``.

    Returns
    -------
    RichResult
        ``ell`` (integer block length for ``method``), ``b_sb``,
        ``b_cb`` (the two real-valued optima), ``G_hat``, ``g0``,
        ``D_sb``, ``D_cb``, ``m_hat``, ``M``, ``n``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 4:
        raise ValueError("boot_block_length_pr: need at least four observations")
    if method not in ("circular", "stationary", "moving"):
        raise ValueError("boot_block_length_pr: method must be circular, stationary or moving")
    cc = float(c)
    if cc <= 0.0:
        raise ValueError("boot_block_length_pr: c must be positive")
    xb = core.mean(xx)
    kmax = n - 1
    R = [0.0] * (kmax + 1)
    for k in range(kmax + 1):
        s = 0.0
        for i in range(n - k):
            s += (xx[i] - xb) * (xx[i + k] - xb)
        R[k] = s / n
    if R[0] <= 0.0:
        raise ValueError("boot_block_length_pr: the series has zero variance")
    # correlogram rule, p.59 footnote c
    KN = int(max(5.0, math.sqrt(math.log10(float(n)))))
    thr = cc * math.sqrt(math.log10(float(n)) / n)
    if m_max is None:
        m_max = int(math.ceil(math.sqrt(float(n)))) + KN
    m_max = int(m_max)
    mhat = 1
    found = False
    for m in range(1, min(m_max, kmax) + 1):
        okm = True
        for kk in range(1, KN + 1):
            idx = m + kk
            if idx > kmax:
                break
            if abs(R[idx] / R[0]) >= thr:
                okm = False
                break
        if okm:
            mhat = m
            found = True
            break
    if not found:
        mhat = min(m_max, kmax)
    M = 2 * mhat
    if M > kmax:
        M = kmax
    # equation (8)
    G = 0.0
    for k in range(-M, M + 1):
        G += _lam(k / float(M)) * abs(k) * R[abs(k)]
    def ghat(w):
        s = 0.0
        for k in range(-M, M + 1):
            s += _lam(k / float(M)) * R[abs(k)] * math.cos(w * k)
        return s
    g0 = ghat(0.0)
    # (2/pi) int_{-pi}^{pi} (1 + cos w) ghat^2(w) dw, fixed trapezoid grid
    h = 2.0 * math.pi / _NPANEL
    acc = 0.0
    for i in range(_NPANEL + 1):
        w = -math.pi + i * h
        gv = ghat(w)
        v = (1.0 + math.cos(w)) * gv * gv
        acc += v * (0.5 if (i == 0 or i == _NPANEL) else 1.0)
    integ = acc * h
    D_sb = 4.0 * g0 * g0 + (2.0 / math.pi) * integ
    D_cb = (4.0 / 3.0) * g0 * g0
    def bopt(D):
        if G == 0.0 or D <= 0.0:
            return 1.0
        return (2.0 * G * G / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_sb = bopt(D_sb)
    b_cb = bopt(D_cb)
    b = b_sb if method == "stationary" else b_cb
    ell = int(math.floor(b + 0.5))
    if ell < 1:
        ell = 1
    if ell > n:
        ell = n
    return RichResult(
        title="Politis-White automatic block length",
        summary_lines=[("n", n), ("m_hat", mhat), ("M", M), ("ell", ell)],
        payload={
            "ell": ell,
            "b_sb": b_sb,
            "b_cb": b_cb,
            "G_hat": G,
            "g0": g0,
            "D_sb": D_sb,
            "D_cb": D_cb,
            "m_hat": mhat,
            "M": M,
            "n": n,
            "estimate": float(ell),
            "method": "Politis and White (2004) Econometric Reviews 23(1):53-70, eqs. (8), (9), (13), (14)",
        },
    )


def cheatsheet():
    return "btblen: b = (2G^2/D)^(1/3) n^(1/3) from a flat-top spectral estimate; the rate is n^(1/3), not n^(1/5)"
