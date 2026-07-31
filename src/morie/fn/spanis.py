"""Geometric anisotropy: correcting direction dependence by a linear map."""

import numpy as np

from ._richresult import RichResult
from ._schab_vario import empirical_semivariogram

__all__ = ["schabenberger_geometric_anisotropy"]


def schabenberger_geometric_anisotropy(coords, z, A_matrix=None, n_bins=15,
                                       max_dist=None):
    r"""
    Correct geometric anisotropy by an affine map of the coordinates.

    Following Matern (1986, p. 19), if :math:`Z_1(s)` is stationary with
    isotropic covariance :math:`C_1`, then :math:`Z(s) = Z_1(Bs)` has

    .. math::

        C(h) = C_1(\|Bh\|)

    which is geometrically anisotropic -- its iso-correlation contours
    are ellipses rather than circles. The transformation is reversed by
    :math:`s^{*} = As` with :math:`A = B^{-1}`, and :math:`Z(s^{*})` is
    isotropic again.

    This function applies ``A_matrix`` to the coordinates and returns the
    empirical semivariogram in the corrected space, alongside the
    uncorrected one so the improvement is visible rather than asserted.

    Parameters
    ----------
    coords : array-like
        Coordinates, shape ``(n, d)``.
    z : array-like
        Observed values, shape ``(n,)``.
    A_matrix : array-like, optional
        The ``(d, d)`` correction :math:`A = B^{-1}`. Defaults to the
        identity, which leaves the coordinates untouched.
    n_bins : int, default 15
        Lag bins for the empirical semivariograms.
    max_dist : float, optional
        Largest lag retained.

    Returns
    -------
    RichResult
        ``lag`` / ``gamma`` / ``n_pairs`` in the corrected space,
        ``gamma_raw`` in the original space, and ``coords_corrected``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 4.3.7, p. 151.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    d = coords.shape[1]
    A = np.eye(d) if A_matrix is None else np.asarray(A_matrix, dtype=float)
    if A.shape != (d, d):
        raise ValueError(f"`A_matrix` must be ({d}, {d}) to match `coords`")
    if abs(np.linalg.det(A)) < 1e-300:
        raise ValueError("`A_matrix` is singular; it must be invertible "
                         "(A = B^-1 for the anisotropy map B)")

    star = coords @ A.T
    lag, gam, cnt = empirical_semivariogram(star, z, n_bins, max_dist)
    _, gam_raw, _ = empirical_semivariogram(coords, z, n_bins, max_dist)
    return RichResult(
        title="Geometric anisotropy correction",
        summary_lines=[("det(A)", float(np.linalg.det(A))),
                       ("bins", int(n_bins))],
        payload={"lag": lag, "gamma": gam, "n_pairs": cnt,
                 "gamma_raw": gam_raw, "coords_corrected": star,
                 "A_matrix": A},
    )


def cheatsheet():
    return "spanis: geometric anisotropy, C(h)=C1(||Bh||), corrected by A=B^-1."
