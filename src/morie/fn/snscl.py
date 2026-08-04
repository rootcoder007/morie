# morie.fn -- function file (rootcoder007/morie)
"""Sn robust scale estimator of Rousseeuw and Croux.

Source: Rousseeuw, P. J. and Croux, C. (1993), "Alternatives to the
median absolute deviation", *Journal of the American Statistical
Association* 88(424):1273-1283.  The JASA article is paywalled and was
NOT read directly.  The estimator was taken from the authors' own
reference implementation, the R package **robustbase** (Rousseeuw is an
author of that package), file ``R/qnsn.R``:

    Sn = c * lomed_i { himed_j |x_i - x_j| },   c = 1.1926

where, over m values, ``himed`` is order statistic m//2 + 1 and
``lomed`` is order statistic (m + 1)//2 (both 1-based).  The inner
median runs over ALL j including j = i, so each inner set contains the
zero |x_i - x_i|; that is what the reference implementation does and it
changes the answer, so it is reproduced exactly rather than tidied.

Unlike Qn, Sn takes no median of a symmetrised set and needs no
location estimate: it is location-free by construction.

The finite-sample bias correction is quoted verbatim from ``qnsn.R``:

    n <= 9:  multiply by
        c(0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131)[n - 1]
    n odd,  n >= 11:  multiply by n / (n - 0.9)
    n even, n >= 10:  multiply by 1

Both medians are taken by a full sort, so the arithmetic path is
identical in Python and R.
"""

from ._richresult import RichResult

__all__ = ["sn_scale"]

_SN_SMALL = [0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131]


def _himed(v):
    """Order statistic m//2 + 1 of an already sorted list (1-based)."""
    return v[len(v) // 2]


def _lomed(v):
    """Order statistic (m+1)//2 of an already sorted list (1-based)."""
    return v[(len(v) + 1) // 2 - 1]


def sn_scale(y, constant=1.1926, finite_corr=True):
    """Sn scale: the low median of the high medians of |x_i - x_j|.

    Parameters
    ----------
    y : array-like
        Sample, at least two finite values.
    constant : float
        Consistency constant.  Default 1.1926.
    finite_corr : bool
        Apply the robustbase finite-sample bias correction.

    Returns
    -------
    RichResult
        ``estimate``, ``raw`` (before the constant and the correction),
        ``inner`` (the n high medians), ``correction``, ``n``.
    """
    x = [float(v) for v in y]
    n = len(x)
    if n < 2:
        raise ValueError("Sn needs at least two observations")
    inner = []
    for i in range(n):
        xi = x[i]
        row = []
        for j in range(n):
            dv = xi - x[j]
            row.append(dv if dv >= 0.0 else -dv)
        row.sort()
        inner.append(_himed(row))
    srt = sorted(inner)
    raw = _lomed(srt)
    est = float(constant) * raw
    if finite_corr:
        if n <= 9:
            corr = _SN_SMALL[n - 2]
        elif n % 2:
            corr = n / (n - 0.9)
        else:
            corr = 1.0
    else:
        corr = 1.0
    est = est * corr
    return RichResult(payload={
        "estimate": float(est), "raw": float(raw),
        "inner": [float(v) for v in inner], "correction": float(corr),
        "constant": float(constant), "n": n,
        "method": "Rousseeuw & Croux (1993) Sn, robustbase qnsn.R definition"})


def cheatsheet():
    return "snscl: Rousseeuw & Croux (1993) Sn robust scale"
