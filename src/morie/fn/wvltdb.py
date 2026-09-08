# morie.fn -- function file (rootcoder007/morie)
"""Multi-level Daubechies wavelet decomposition of a periodic signal.

SOURCE.  Daubechies, I. (1992), *Ten Lectures on Wavelets*, CBMS-NSF
Regional Conference Series in Applied Mathematics 61, SIAM;
doi:10.1137/1.9781611970104.  Chapter 5 is the multiresolution analysis
and the pyramid (Mallat) algorithm it implies: with the scaling filter h
and its quadrature mirror g_k = (-1)^k h_{L-1-k}, one level of the
periodic decomposition of a length-M sequence is

    a_{j+1}[i] = sum_k h_k a_j[(2i + k) mod M]
    d_{j+1}[i] = sum_k g_k a_j[(2i + k) mod M],   i = 0 ... M/2 - 1,

repeated on the approximation a_{j+1}.  The filters themselves are the
ones constructed in Daubechies (1988),
doi:10.1002/cpa.3160410705, and are taken from
:mod:`morie.fn.wave` rather than re-derived here; that module also
carries the algebraic checks on them.

The synthesis step is the exact adjoint of the analysis step,

    a_j[(2i + k) mod M] += h_k a_{j+1}[i] + g_k d_{j+1}[i],

which reconstructs exactly because the analysis operator is orthonormal.
``reconstruction_error`` in the payload is the max |x - synth(analyse
x))|; it is an anchor on the transform that does not depend on the R arm
agreeing.

SCOPE.  Periodic (circular) boundary handling only, and signal length a
power of two.  Both are this implementation's scope choices, stated
rather than attributed.
"""

from __future__ import annotations

from . import _s03core as core
from .wave import _dbfilter, _pow2, _step as _dbstep_public

from ._richresult import RichResult

__all__ = ["db_wavelet"]


def _synth(a, d, h, g):
    m = len(a)
    n = 2 * m
    L = len(h)
    out = [0.0] * n
    for i in range(m):
        for k in range(L):
            out[(2 * i + k) % n] += h[k] * a[i] + g[k] * d[i]
    return out


def db_wavelet(y, level=None, wavelet="db2"):
    """Multi-level periodic DWT with a Daubechies filter.

    Parameters
    ----------
    y : array-like
        Signal of length 2^J, J >= 1.
    level : int or None
        Number of decomposition levels; default the full J.
    wavelet : str
        ``"db1"``, ``"db2"`` or ``"db3"``.

    Returns
    -------
    RichResult
        ``details`` (list of detail vectors, coarsest last),
        ``approximation`` (final a_J), ``coefficients`` (the packed
        vector [a_J, d_J, ..., d_1]), ``energies`` (per-level detail
        energy), ``approximation_energy``, ``reconstruction``,
        ``reconstruction_error``, ``h``, ``g``, ``n``, ``level``.

    Raises
    ------
    ValueError
        Length not a power of two, level out of range, unknown wavelet,
        or a filter longer than the coarsest level.

    References
    ----------
    Daubechies, I. (1992).  Ten Lectures on Wavelets.  SIAM.
    doi:10.1137/1.9781611970104.
    Daubechies, I. (1988).  Communications on Pure and Applied
    Mathematics 41(7):909-996.  doi:10.1002/cpa.3160410705.
    """
    x = core.vec(y)
    n = len(x)
    J = _pow2(n)
    if J < 1:
        raise ValueError("db_wavelet: length of y must be a power of two, at least 2")
    lev = J if level is None else int(level)
    if lev < 1 or lev > J:
        raise ValueError("db_wavelet: level must lie in 1 .. log2(n)")
    h = _dbfilter(wavelet)
    g = [((-1.0) ** k) * h[len(h) - 1 - k] for k in range(len(h))]
    if len(h) > (n >> (lev - 1)):
        raise ValueError("db_wavelet: filter is longer than the coarsest level")
    a = list(x)
    details = []
    for _ in range(lev):
        a, d = _dbstep_public(a, h, g)
        details.append(d)
    packed = list(a)
    for d in reversed(details):
        packed = packed + d
    rec = list(a)
    for j in range(lev - 1, -1, -1):
        rec = _synth(rec, details[j], h, g)
    err = 0.0
    for i in range(n):
        e = abs(rec[i] - x[i])
        if e > err:
            err = e
    energies = [sum(v * v for v in d) for d in details]
    ea = sum(v * v for v in a)
    return RichResult(
        title="Daubechies multi-level wavelet decomposition",
        summary_lines=[("n", n), ("wavelet", wavelet), ("levels", lev)],
        payload={
            "estimate": ea,
            "details": details,
            "approximation": a,
            "coefficients": packed,
            "energies": energies,
            "approximation_energy": ea,
            "reconstruction": rec,
            "reconstruction_error": err,
            "h": h,
            "g": g,
            "n": n,
            "level": lev,
            "method": "Periodic pyramid (Mallat) multiresolution decomposition, Daubechies (1992) Ch. 5",
        },
    )


def cheatsheet():
    return "wvltdb: multi-level Daubechies wavelet decomposition (Daubechies 1992 Ch. 5)"

# public names resolved by fn/_lazy_map.json
dbwavelet = db_wavelet
