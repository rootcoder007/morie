# morie.fn -- slice s03 (rootcoder007/morie)
"""RMSNorm, root-mean-square layer normalisation.

Source consulted (FETCHED): Zhang, B. and Sennrich, R. (2019).  Root
mean square layer normalization.  *NeurIPS* 32 (arXiv:1910.07467),
equation (4):

    abar_i = a_i / RMS(a) * g_i,   RMS(a) = sqrt( (1/n) sum_i a_i^2 )

to be contrasted with LayerNorm's equation (2), abar_i = (a_i - mu)/sigma
* g_i: RMSNorm drops the re-centering entirely, which is the paper's
whole hypothesis -- "re-centering invariance in LayerNorm is
dispensable".  The paper also proposes pRMSNorm, in which the RMS is
estimated from the first p per cent of the units; that variant is
available here as ``p``, and it is *not* the default, because it changes
the statistic.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["rms_norm"]


def rms_norm(y, x=None, g=None, eps=0.0, p=1.0, b=None):
    """RMSNorm of a vector.

    Parameters
    ----------
    y : array-like
        The summed inputs a.  (First slot, for signature stability.)
    x : array-like, optional
        The summed inputs; wins over ``y`` when given.
    g : array-like, optional
        The gain; ones by default, as the paper initialises it.
    eps : float
        Added inside the square root for numerical safety; 0 gives the
        paper's exact expression.
    p : float
        Fraction of units used to estimate the RMS (pRMSNorm); 1 is the
        full RMS and is the default.
    b : array-like, optional
        Offset added after scaling.

    Returns
    -------
    RichResult with payload:
        estimate : the first normalised unit
        out      : the normalised vector
        rms      : the RMS actually used
        k_partial: number of units the RMS was estimated from
    """
    a = k.vec(x if x is not None else y)
    n = len(a)
    kp = int(n * float(p))
    if kp < 1:
        kp = 1
    if kp > n:
        kp = n
    s = 0.0
    for i in range(kp):
        s += a[i] * a[i]
    rms = math.sqrt(s / kp + float(eps))
    gg = k.vec(g) if g is not None else [1.0] * n
    bb = k.vec(b) if b is not None else [0.0] * n
    out = [(a[i] / rms) * gg[i] + bb[i] if rms > 0.0 else 0.0 for i in range(n)]
    return RichResult(
        title="RMSNorm",
        summary_lines=[("RMS", rms), ("units", n)],
        payload={
            "estimate": out[0] if out else float("nan"),
            "out": out,
            "rms": rms,
            "k_partial": kp,
            "n": n,
            "method": "RMSNorm (Zhang and Sennrich 2019, eq. 4)",
        },
    )


def cheatsheet():
    return "rmsnrm: RMSNorm -- root-mean-square layer normalization"
