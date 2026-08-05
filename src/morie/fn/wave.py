# morie.fn -- function file (rootcoder007/morie)
"""Daubechies orthonormal wavelet basis for a periodic signal.

SOURCE.  Daubechies, I. (1988), "Orthonormal bases of compactly
supported wavelets", *Communications on Pure and Applied Mathematics*
41(7):909-996, doi:10.1002/cpa.3160410705.

The scaling filter h of length L = 2N is the object the paper
constructs: the compactly supported solution of the two-scale relation
whose defining conditions are

    sum_k h_k        = sqrt(2)                       (normalisation)
    sum_k h_k h_{k+2m} = delta_{m0}                  (double-shift
                                                      orthonormality)
    sum_k (-1)^k k^m h_k = 0,  m = 0 ... N-1         (N vanishing
                                                      moments)

and the wavelet filter is the quadrature mirror g_k = (-1)^k h_{L-1-k}.
Those three conditions determine h up to reflection, and they are
checked numerically here rather than taken on trust -- ``orthonormality``,
``double_shift`` and ``vanishing_moments`` in the payload are the
residuals, and every one of them is an anchor that does not run through
the transform.

FILTERS PROVIDED.  ``db1`` (Haar, N = 1), ``db2`` (N = 2) and ``db3``
(N = 3), each written from its exact radical closed form rather than
from a decimal table:

    db1:  h = (1, 1) / sqrt(2)
    db2:  h = (1+r3, 3+r3, 3-r3, 1-r3) / (4 sqrt(2)),   r3 = sqrt(3)
    db3:  h = c / sqrt(2) with 16 c = (1+a+b, 5+a+3b, 10-2a+2b,
          10-2a-2b, 5+a-3b, 1+a-b),  a = sqrt(10), b = sqrt(5+2a).

Longer filters have no such compact radical form and are NOT provided;
that restriction is this implementation's scope choice, stated rather
than attributed.  The db3 closed form above is verified algebraically by
the payload residuals (sum c^2 = 2 exactly, sum c = 2 exactly).

BASIS.  For a signal of length N = 2^J the periodic pyramid transform is
an orthonormal map, and its matrix W is returned.  It is built by
transforming the standard basis vectors, so W is by construction the
matrix of the transform actually used, and W W' = I is then a real check
on the pyramid code rather than a tautology.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["wavelet_basis"]


def _dbfilter(name):
    """Scaling filter h from the exact radical closed form."""
    key = str(name).strip().lower()
    r2 = math.sqrt(2.0)
    if key in ("db1", "haar", "d2", "1"):
        return [1.0 / r2, 1.0 / r2]
    if key in ("db2", "d4", "2"):
        r3 = math.sqrt(3.0)
        s = 4.0 * r2
        return [(1.0 + r3) / s, (3.0 + r3) / s, (3.0 - r3) / s, (1.0 - r3) / s]
    if key in ("db3", "d6", "3"):
        a = math.sqrt(10.0)
        b = math.sqrt(5.0 + 2.0 * a)
        c = [(1.0 + a + b) / 16.0, (5.0 + a + 3.0 * b) / 16.0,
             (10.0 - 2.0 * a + 2.0 * b) / 16.0, (10.0 - 2.0 * a - 2.0 * b) / 16.0,
             (5.0 + a - 3.0 * b) / 16.0, (1.0 + a - b) / 16.0]
        return [v / r2 for v in c]
    raise ValueError("wavelet_basis: unknown wavelet %r (db1, db2, db3)" % (name,))


def _mirror(h):
    L = len(h)
    return [((-1.0) ** k) * h[L - 1 - k] for k in range(L)]


def _step(a, h, g):
    """One periodic pyramid step: returns (approximation, detail)."""
    n = len(a)
    m = n // 2
    L = len(h)
    ap = [0.0] * m
    de = [0.0] * m
    for i in range(m):
        sa = 0.0
        sd = 0.0
        for k in range(L):
            v = a[(2 * i + k) % n]
            sa += h[k] * v
            sd += g[k] * v
        ap[i] = sa
        de[i] = sd
    return ap, de


def _forward(x, h, g, level):
    a = list(x)
    coeffs = []
    for _ in range(level):
        a, d = _step(a, h, g)
        coeffs.append(d)
    out = list(a)
    for d in reversed(coeffs):
        out = out + d
    return out


def _pow2(n):
    k = 0
    m = n
    while m > 1:
        if m % 2:
            return -1
        m //= 2
        k += 1
    return k if n >= 1 else -1


def wavelet_basis(y, wavelet="db2", level=None):
    """Orthonormal Daubechies wavelet basis and the transform of ``y``.

    Parameters
    ----------
    y : array-like
        Signal of length 2^J, J >= 1.
    wavelet : str
        ``"db1"``, ``"db2"`` or ``"db3"``.
    level : int or None
        Pyramid depth; default the full J levels.

    Returns
    -------
    RichResult
        ``basis`` (the N-by-N matrix W, rows = basis vectors),
        ``coefficients`` (W y), ``h``, ``g``, ``orthonormality``
        (max |sum h^2 - 1|), ``double_shift`` (max |sum_k h_k h_{k+2m}|,
        m >= 1), ``vanishing_moments`` (max |sum_k (-1)^k k^m h_k|,
        m < N), ``gram_error`` (max |W W' - I|), ``energy_in``,
        ``energy_out``, ``n``, ``level``, ``filter_length``.

    Raises
    ------
    ValueError
        Unknown wavelet, length not a power of two, level out of range,
        or a filter longer than the signal.

    References
    ----------
    Daubechies, I. (1988).  Communications on Pure and Applied
    Mathematics 41(7):909-996.  doi:10.1002/cpa.3160410705.
    """
    x = core.vec(y)
    n = len(x)
    J = _pow2(n)
    if J < 1:
        raise ValueError("wavelet_basis: length of y must be a power of two, at least 2")
    h = _dbfilter(wavelet)
    L = len(h)
    g = _mirror(h)
    lev = J if level is None else int(level)
    if lev < 1 or lev > J:
        raise ValueError("wavelet_basis: level must lie in 1 .. log2(n)")
    if L > n:
        raise ValueError("wavelet_basis: filter is longer than the signal")
    # defining conditions of Daubechies (1988), checked not assumed
    orth = abs(sum(v * v for v in h) - 1.0)
    ds = 0.0
    m = 1
    while 2 * m < L:
        s = 0.0
        for k in range(L - 2 * m):
            s += h[k] * h[k + 2 * m]
        ds = max(ds, abs(s))
        m += 1
    vm = 0.0
    for p in range(L // 2):
        s = 0.0
        for k in range(L):
            s += ((-1.0) ** k) * (float(k) ** p if p > 0 else 1.0) * h[k]
        vm = max(vm, abs(s))
    nrm = abs(sum(h) - math.sqrt(2.0))
    W = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        W.append(_forward(e, h, g, lev))
    # W as built has columns indexed by i; transpose so rows are basis vectors
    W = core.tr(W)
    gram = 0.0
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += W[i][k] * W[j][k]
            gram = max(gram, abs(s - (1.0 if i == j else 0.0)))
    co = core.matvec(W, x)
    ein = sum(v * v for v in x)
    eout = sum(v * v for v in co)
    return RichResult(
        title="Daubechies orthonormal wavelet basis",
        summary_lines=[("n", n), ("wavelet", wavelet), ("levels", lev)],
        payload={
            "estimate": eout,
            "basis": W,
            "coefficients": co,
            "h": h,
            "g": g,
            "orthonormality": orth,
            "normalisation": nrm,
            "double_shift": ds,
            "vanishing_moments": vm,
            "gram_error": gram,
            "energy_in": ein,
            "energy_out": eout,
            "n": n,
            "level": lev,
            "filter_length": L,
            "method": "Daubechies (1988) orthonormal wavelet basis, periodic pyramid",
        },
    )


def cheatsheet():
    return "wave: Daubechies orthonormal wavelet basis (db1/db2/db3), Daubechies (1988)"
