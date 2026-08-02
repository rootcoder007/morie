# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parametric non-stationary correlation: the point-source model."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_nonstat import point_source_correlation, practical_range

__all__ = ["schabenberger_nonstationary_cov"]


def schabenberger_nonstationary_cov(coords, z=None, source=None, theta1=1.0,
                                    theta2=0.0, theta3=0.0, sill=1.0,
                                    anisotropy=None, source_anisotropy=None):
    """The point-source correlation model of Sec. 8.2.1, eq (8.1).

    ``Corr[Z(s_i),Z(s_j)] = exp{-theta1 ||s_i-s_j||
    exp{theta2 |c_i-c_j| + theta3 min[c_i,c_j]}}``

    where ``c_i`` is the distance from site i to the point source. The model
    is non-stationary because the correlation of a pair depends on where the
    pair sits relative to the source, not only on their separation.

    With ``theta2 = theta3 = 0`` it collapses to the exponential correlation
    model with practical range ``3/theta1``. In general the pair behaves like
    an exponential model with practical range
    ``3 exp{-theta2|c_i-c_j| - theta3 min[c_i,c_j]} / theta1``.

    Sec. 8.2.1 states that ``theta1 > 0`` and ``theta2, theta3 >= 0`` are
    necessary but *not* sufficient for positive semi-definiteness, and that
    the eigenvalues must be examined. ``min_eigenvalue`` and ``valid`` are
    therefore always returned, and an invalid model is flagged rather than
    passed on silently. A search over the parameter space finds such cases
    readily: n=20, theta1=0.021, theta2=0.457, theta3=0.037 gives a minimum
    eigenvalue of -0.146 despite satisfying every stated constraint.

    The stub this replaces printed ``C(s1,s2) = sigma(s1) sigma(s2)
    rho(s1,s2)``, a generic heteroscedastic form that does not appear in
    Sec. 8.2.1.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    z : ignored; the model is a covariance structure, not a fit
    source : array-like, shape (d,)
        The point source ``c``. Defaults to the centroid of ``coords``.
    theta1 : float > 0
    theta2, theta3 : float >= 0
    sill : float, default 1.0
        Scales the correlation into a covariance.
    anisotropy, source_anisotropy : (d, d) arrays, optional
        ``A`` and ``A_c`` of p. 423.

    Returns
    -------
    RichResult
        Keys: ``nonstationary_cov``, ``correlation``, ``source_distance``,
        ``separation``, ``min_eigenvalue``, ``valid``, ``practical_range``,
        ``theta``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005), Sec. 8.2.1, eq (8.1),
    pp. 422-423. Hughes-Oliver, J. M., Gonzalez-Farias, G., Lu, J.-C. & Chen,
    D. (1998a), Parametric nonstationary correlation models, Statistics &
    Probability Letters 40:267-278.
    """
    s = np.asarray(coords, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if source is None:
        source = s.mean(axis=0)
    res = point_source_correlation(s, source, theta1, theta2, theta3,
                                   anisotropy=anisotropy,
                                   source_anisotropy=source_anisotropy)
    sill = float(sill)
    if sill <= 0:
        raise ValueError("sill must be positive")
    payload = dict(res)
    payload["nonstationary_cov"] = sill * res["correlation"]
    payload["sill"] = sill
    payload["practical_range"] = practical_range(theta1)
    payload["source"] = np.asarray(source, dtype=float)
    lines = [("theta", res["theta"]),
             ("practical range (theta2=theta3=0)", payload["practical_range"]),
             ("min eigenvalue", res["min_eigenvalue"]),
             ("positive semi-definite", res["valid"])]
    return RichResult(title="Point-source non-stationary correlation",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spnst: Hughes-Oliver point-source non-stationary correlation "
            "(Sec. 8.2.1, eq (8.1)) with the required eigenvalue check")
