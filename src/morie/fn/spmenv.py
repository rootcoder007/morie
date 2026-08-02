# SPDX-License-Identifier: AGPL-3.0-or-later
"""Moments of Moran's I under Gaussianity and under randomization."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_moran import geary_c, moran_i, moran_moments, weight_sums

__all__ = ["schabenberger_moran_expectation"]


def schabenberger_moran_expectation(x, w):
    """Mean and variance of Moran's I, Sec. 1.3.2.

    Both assumptions give the same mean, ``E[I] = -1/(n-1)`` (p. 22), but the
    variances differ and this returns both:

    ``variance_normal``
        ``(n^2 S1 - n S2 + 3 S0^2) / {S0^2 (n^2-1)} - E[I]^2``, the variance
        when the ``Z(s_i)`` are treated as Gaussian. It does not involve the
        data at all -- only the weights and ``n``.
    ``variance_randomization``
        Problem 1.8's ``E_r[I^2]`` less ``E[I]^2``. This one does depend on
        the data, through the sample kurtosis ``b``.

    Verified against Example 1.7 (10x10 rook lattice): the normality
    standard deviation comes out at 0.0731, the value the book prints.

    Two corrections were needed against the stub this replaces. It printed
    ``(n^2 S1 - n S2 + 3 S0^2)/(S0^2(n^2-1))`` and labelled it the
    randomization variance; that expression is the *normality* variance, and
    it was missing the ``- E[I]^2`` term. Separately, the ``E_r[I^2]``
    printed in Problem 1.8 is missing a bracket -- ``n`` multiplies the whole
    first group, which is what the book's own Example 1.7 confirms.

    Parameters
    ----------
    x : array-like, shape (n,)
        Attribute values on the lattice.
    w : array-like, shape (n, n)
        Spatial connectivity weights, zero diagonal.

    Returns
    -------
    RichResult
        Keys: ``I``, ``expectation``, ``variance_normal``,
        ``variance_randomization``, ``sd_normal``, ``sd_randomization``,
        ``z_normal``, ``z_randomization``, ``kurtosis_b``, ``S0``, ``S1``,
        ``S2``, ``geary_c``, ``geary_expectation``, ``n``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for Spatial
    Data Analysis. Chapman & Hall/CRC. Sec. 1.3.2, eqs (1.14)-(1.15),
    pp. 21-23; Example 1.7 p. 22; Problem 1.8 p. 39, which quotes Cliff, A. D.
    & Ord, J. K. (1981), Spatial Processes: Models and Applications, Pion,
    Ch. 2 / p. 21.
    """
    m = moran_moments(x, w)
    lines = [
        ("I", m["I"]),
        ("E[I]", m["expectation"]),
        ("sd (Gaussian)", m["sd_normal"]),
        ("sd (randomization)", m["sd_randomization"]),
        ("Z (Gaussian)", m["z_normal"]),
        ("Z (randomization)", m["z_randomization"]),
    ]
    return RichResult(title="Moments of Moran's I", summary_lines=lines,
                      payload=dict(m, moments=m))


def cheatsheet():
    return ("spmenv: E[I] and Var[I] for Moran's I under Gaussianity and "
            "under randomization (Sec. 1.3.2, Problem 1.8)")
