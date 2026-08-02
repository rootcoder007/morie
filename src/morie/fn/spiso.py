"""Isotropy: covariance depends on lag distance, not direction."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_vario import empirical_semivariogram

__all__ = ["schabenberger_isotropy_condition"]


def schabenberger_isotropy_condition(coords, z, n_dir=4, n_bins=10,
                                     max_dist=None, tol=0.25):
    r"""
    Isotropy check by comparing directional semivariograms.

    A second-order stationary field is ISOTROPIC when the covariance
    depends only on the LENGTH of the lag,

    .. math::  C(h) = C(\|h\|)

    so no direction is special. Anisotropy means the dependence differs
    by direction: under geometric anisotropy the iso-correlation contours
    are ellipses rather than circles, and the fix is the affine
    correction of Sec. 4.3.7.

    The check splits pairs into ``n_dir`` angular sectors on the half
    circle (a lag and its negation are the same direction), fits a
    semivariogram in each, and compares them. The statistic is the
    spread across directions relative to the overall level; isotropy is
    rejected when that exceeds ``tol``.

    Parameters
    ----------
    coords : array-like
        Coordinates, shape ``(n, 2)``.
    z : array-like
        Observed values, shape ``(n,)``.
    n_dir : int, default 4
        Angular sectors on the half circle.
    n_bins : int, default 10
        Lag bins within each sector.
    max_dist : float, optional
        Largest lag retained.
    tol : float, default 0.25
        Relative spread above which isotropy is rejected.

    Returns
    -------
    RichResult
        ``is_isotropic``, ``relative_spread``, ``directional_gamma``
        (one row per sector), ``angles``, ``omnidirectional``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 2.2; the anisotropy
    correction is Sec. 4.3.7.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    if coords.shape[0] != z.size:
        raise ValueError("`coords` and `z` must have the same number of rows")
    if coords.shape[1] != 2:
        raise ValueError("directional analysis needs 2-D `coords`")
    n_dir = int(n_dir)
    if n_dir < 2:
        raise ValueError("`n_dir` must be >= 2")

    i, j = np.triu_indices(z.size, k=1)
    d = coords[j] - coords[i]
    dist = np.linalg.norm(d, axis=1)
    # A lag and its negation are the same direction, so the angle folds onto
    # [0, pi). Orient every lag into one half-space BEFORE atan2 rather than
    # folding after: for a pair enumerated in the opposite order atan2 returns
    # the supplement, and folding that back mod pi lands one ulp away from the
    # direct value. That is invisible except for lags sitting exactly on a
    # sector boundary -- on a regular lattice thousands of them do -- where it
    # silently reassigns the pair to the neighbouring direction and makes the
    # result depend on the order the points were listed in.
    flip = (d[:, 0] < 0.0) | ((d[:, 0] == 0.0) & (d[:, 1] < 0.0))
    d = np.where(flip[:, None], -d, d)
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)
    sq = (z[i] - z[j]) ** 2
    if max_dist is None:
        max_dist = dist.max() / 2.0 if dist.size else 1.0

    edges = np.linspace(0.0, np.pi, n_dir + 1)
    # A lag whose true direction lies exactly on a sector boundary lands a few
    # ulp either side of the edge: atan2 is not correctly rounded and differs
    # between platforms' libm, and the lattice offsets themselves are not
    # exact in binary. On a regular grid that is not a rare event -- thousands
    # of pairs sit on the diagonals -- so letting the last bit decide the
    # sector makes the answer platform-dependent. Snap to the edge at the
    # resolution of the angle computation and let the half-open
    # [edge_a, edge_a+1) convention place them.
    ang_tol = 8.0 * float(np.spacing(np.pi))
    for e in edges:
        ang = np.where(np.abs(ang - e) < ang_tol, e, ang)
    ang = np.where(np.abs(ang - np.pi) < ang_tol, 0.0, ang)
    lagedges = np.linspace(0.0, max_dist, n_bins + 1)
    gam = np.full((n_dir, n_bins), np.nan)
    for a in range(n_dir):
        sel = (ang >= edges[a]) & (ang < edges[a + 1]) & (dist <= max_dist)
        if not np.any(sel):
            continue
        b = np.clip(np.digitize(dist[sel], lagedges) - 1, 0, n_bins - 1)
        for k in range(n_bins):
            m = b == k
            if m.sum():
                gam[a, k] = sq[sel][m].sum() / (2.0 * m.sum())

    _, omni, _ = empirical_semivariogram(coords, z, n_bins, max_dist)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(gam, axis=0) - np.nanmin(gam, axis=0)
        level = np.nanmean(gam, axis=0)
        rel = np.nanmean(spread / np.where(level > 0, level, np.nan))
    rel = float(rel) if np.isfinite(rel) else float("nan")
    return RichResult(
        title="Isotropy check",
        summary_lines=[("directions", n_dir), ("relative spread", rel)],
        payload={"is_isotropic": bool(np.isfinite(rel) and rel <= tol),
                 "relative_spread": rel, "directional_gamma": gam,
                 "angles": (edges[:-1] + edges[1:]) / 2.0,
                 "omnidirectional": omni, "tol": float(tol)},
    )


def cheatsheet():
    return "spiso: isotropic iff C depends on ||h|| alone; directional check."
