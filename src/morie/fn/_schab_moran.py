# SPDX-License-Identifier: AGPL-3.0-or-later
"""Moments of Moran's I and Geary's c on a lattice.

Schabenberger & Gotway (2005), Sec. 1.3.2, pp. 21-23.

  eq (1.14)  I = n / {(n-1) S^2 w..} sum_i sum_j w_ij (Z_i - Zbar)(Z_j - Zbar)
  eq (1.15)  c = 1 / {2 S^2 w..}     sum_i sum_j w_ij (Z_i - Z_j)^2

with ``S^2`` the sample variance (divisor n-1) and ``w.. = sum_i sum_j w_ij``.
Substituting ``(n-1) S^2 = sum (Z_i - Zbar)^2`` recovers the familiar forms.

Both assumptions give the same means (p. 22),

  E_g[I] = E_r[I] = -1/(n-1),        E_g[c] = E_r[c] = 1,

"but expressions for the variances under Gaussianity and randomization
differ". For those the book refers to Cliff and Ord (1981, p. 21) and to its
own Problem 1.8, which prints ``E_r[I^2]``.

Two things had to be settled before this could be written
----------------------------------------------------------
1. **The printed Problem 1.8 formula is missing a bracket.** As typeset it
   reads ``n[n^2 - 3n + 3]S1 - nS2 + 3w..^2 - b[...]``, i.e. ``n`` multiplying
   only ``S1``. Read that way the numerator is wrong. The grouping that the
   book's own Example 1.7 confirms is ``n[(n^2-3n+3)S1 - nS2 + 3w..^2]``, with
   ``n`` multiplying the whole first group. Measured on the 10x10 rook lattice
   of Example 1.7:

       standard grouping   sd(I) = 0.0732 at b = 2.7   book prints 0.0732
       literal as printed  sd(I) = 0.0740              no match

2. **The stub this module replaces printed the wrong variance entirely.** It
   gave ``(n^2 S1 - n S2 + 3 S0^2) / (S0^2 (n^2-1))`` and called it the
   randomization variance. That expression is the *normality* variance, and
   even for that it is incomplete: the ``- E[I]^2`` term is missing. With the
   term restored it reproduces Example 1.7's Gaussian column exactly
   (sd = 0.0731 against the book's 0.0731).

Both variances are returned, named for their assumption, because the two are
not interchangeable and the book prints them side by side.
"""

from . import _array_core as np

__all__ = [
    "geary_c",
    "moran_i",
    "moran_moments",
    "weight_sums",
]


def _as_weights(w):
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"weights must be a square matrix, got shape {w.shape}")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(np.diag(w) != 0):
        raise ValueError("weights must have a zero diagonal; a site is not its own neighbour")
    return w


def weight_sums(w):
    """The three weight sums every moment formula is built from.

    ``S0 = w.. = sum_i sum_j w_ij``  (the book writes ``w..``)
    ``S1 = (1/2) sum_i sum_j (w_ij + w_ji)^2``
    ``S2 = sum_i (sum_j w_ij + sum_j w_ji)^2``

    Problem 1.8 defines S1 and S2 exactly this way; S0 is the ``w..`` of
    eqs (1.14) and (1.15).
    """
    w = _as_weights(w)
    s0 = float(w.sum())
    if s0 <= 0:
        raise ValueError("total weight w.. must be positive; the lattice has no neighbours")
    s1 = float(0.5 * ((w + w.T) ** 2).sum())
    s2 = float(((w.sum(axis=1) + w.sum(axis=0)) ** 2).sum())
    return {"S0": s0, "S1": s1, "S2": s2}


def moran_i(z, w):
    """Moran's I, eq (1.14)."""
    z = np.asarray(z, dtype=float).ravel()
    w = _as_weights(w)
    n = z.size
    if w.shape[0] != n:
        raise ValueError(f"z has length {n} but the weight matrix is {w.shape[0]}x{w.shape[0]}")
    if n < 4:
        raise ValueError("at least 4 sites are needed; the randomization variance divides by n-3")
    d = z - z.mean()
    s0 = float(w.sum())
    denom = float(d @ d)
    if denom <= 0:
        raise ValueError("z is constant; Moran's I is undefined")
    return float(n * (d @ w @ d) / (s0 * denom))


def geary_c(z, w):
    """Geary's c, eq (1.15)."""
    z = np.asarray(z, dtype=float).ravel()
    w = _as_weights(w)
    n = z.size
    if w.shape[0] != n:
        raise ValueError(f"z has length {n} but the weight matrix is {w.shape[0]}x{w.shape[0]}")
    d = z - z.mean()
    denom = float(d @ d)
    if denom <= 0:
        raise ValueError("z is constant; Geary's c is undefined")
    diff2 = (z[:, None] - z[None, :]) ** 2
    return float((n - 1) * float((w * diff2).sum()) / (2.0 * float(w.sum()) * denom))


def _kurtosis_b(z):
    """``b`` of Problem 1.8: n sum (Z-Zbar)^4 / {sum (Z-Zbar)^2}^2.

    This is the sample kurtosis, and it is the only place the DATA enter the
    randomization variance. Gaussian data give b near 3; the normality
    variance does not involve b at all.
    """
    d = np.asarray(z, dtype=float).ravel()
    d = d - d.mean()
    s2 = float(d @ d)
    if s2 <= 0:
        raise ValueError("z is constant; the kurtosis is undefined")
    return float(d.size * float((d ** 4).sum()) / (s2 ** 2))


def moran_moments(z, w):
    """Mean and variance of Moran's I under both assumptions.

    Returns the observed I, the common mean ``-1/(n-1)``, the variance under
    Gaussianity and under randomization, and the standardised statistic for
    each.

    ``variance_normal``
        ``(n^2 S1 - n S2 + 3 S0^2) / {S0^2 (n^2-1)} - E[I]^2``.
        Verified against Example 1.7: on the 10x10 rook lattice this gives
        sd = 0.0731, the value the book prints.

    ``variance_randomization``
        Problem 1.8's ``E_r[I^2]`` minus ``E[I]^2``, with the first group
        bracketed as the example requires:

            E_r[I^2] = { n[(n^2-3n+3)S1 - n S2 + 3 S0^2]
                         - b[(n^2-n)S1 - 2n S2 + 6 S0^2] }
                       / {(n-1)(n-2)(n-3) S0^2}
    """
    z = np.asarray(z, dtype=float).ravel()
    s = weight_sums(w)
    s0, s1, s2 = s["S0"], s["S1"], s["S2"]
    n = float(z.size)
    if n < 4:
        raise ValueError("at least 4 sites are needed; the randomization variance divides by n-3")

    i_obs = moran_i(z, w)
    e_i = -1.0 / (n - 1.0)

    var_norm = (n * n * s1 - n * s2 + 3.0 * s0 * s0) / (s0 * s0 * (n * n - 1.0)) - e_i * e_i

    b = _kurtosis_b(z)
    first = n * ((n * n - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s0 * s0)
    second = b * ((n * n - n) * s1 - 2.0 * n * s2 + 6.0 * s0 * s0)
    e_i2 = (first - second) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s0 * s0)
    var_rand = e_i2 - e_i * e_i

    def _z(v):
        return (i_obs - e_i) / np.sqrt(v) if v > 0 else np.nan

    return {
        "I": i_obs,
        "expectation": e_i,
        "variance_normal": float(var_norm),
        "variance_randomization": float(var_rand),
        "sd_normal": float(np.sqrt(var_norm)) if var_norm > 0 else np.nan,
        "sd_randomization": float(np.sqrt(var_rand)) if var_rand > 0 else np.nan,
        "z_normal": float(_z(var_norm)),
        "z_randomization": float(_z(var_rand)),
        "kurtosis_b": b,
        "S0": s0,
        "S1": s1,
        "S2": s2,
        "n": int(n),
        # Geary's mean is 1 under both assumptions (p. 22); reported so the
        # caller can see the pair the book presents together.
        "geary_c": geary_c(z, w),
        "geary_expectation": 1.0,
    }
