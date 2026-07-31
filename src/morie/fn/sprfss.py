"""Stationarity definitions: strict, second-order, intrinsic."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_random_field_stationarity"]


def schabenberger_random_field_stationarity(coords, z, n_blocks=4, n_bins=10,
                                            max_dist=None, tol=0.25):
    r"""
    Which stationarity assumption the data can support.

    The book's hierarchy (Sec. 2.2):

    STRICT
        the spatial distribution is invariant under translation,
        :math:`\Pr(Z(s_1)<z_1,\dots) = \Pr(Z(s_1+h)<z_1,\dots)` for all
        :math:`k` and :math:`h`. A strictly stationary field repeats
        itself throughout the domain.
    SECOND-ORDER (weak)
        only the first two moments are required:
        :math:`E[Z(s)] = \mu` constant and
        :math:`\mathrm{Cov}[Z(s), Z(s+h)] = C(h)` depending on the lag
        alone.
    INTRINSIC
        weaker still: only the INCREMENTS need be stationary, so
        :math:`E[Z(s+h)-Z(s)] = 0` and
        :math:`\mathrm{Var}[Z(s+h)-Z(s)] = 2\gamma(h)`. A process can be
        intrinsically stationary with no finite variance and hence no
        covariance function at all.

    Second-order stationarity does NOT imply strict stationarity in
    general -- but it does in a Gaussian random field, where the first
    two moments determine the distribution. That implication is reported
    rather than assumed, since it depends on an assumption about the
    field the data cannot settle.

    What is checked here is the moment conditions: whether the local mean
    and the local variance drift across blocks of the domain. Drift in
    the mean breaks second-order stationarity; drift in the variance
    breaks it too but leaves the increments possibly usable.

    Parameters
    ----------
    coords : array-like
        Coordinates, shape ``(n, d)``.
    z : array-like
        Observed values, shape ``(n,)``.
    n_blocks : int, default 4
        Blocks per axis for the drift check.
    n_bins, max_dist
        Passed through to the increment check.
    tol : float, default 0.25
        Relative drift above which a condition is judged violated.

    Returns
    -------
    RichResult
        ``mean_stationary``, ``variance_stationary``,
        ``second_order_plausible``, ``intrinsic_plausible``,
        ``strict_if_gaussian``, ``mean_drift``, ``variance_drift``,
        ``increment_bias`` (max |E[Z(s+h)-Z(s)]| over lag bins, scaled),
        ``block_means``, ``block_vars``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 2.2, pp. 42-43;
    the Gaussian implication at p. 48; the intrinsic hypothesis at p. 51.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    if coords.shape[0] != z.size:
        raise ValueError("`coords` and `z` must have the same number of rows")
    n_blocks = int(n_blocks)
    if n_blocks < 2:
        raise ValueError("`n_blocks` must be >= 2")

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    idx = np.clip(((coords - lo) / span * n_blocks).astype(int), 0, n_blocks - 1)
    key = idx[:, 0] if coords.shape[1] == 1 else idx[:, 0] * n_blocks + idx[:, 1]
    means, vars_ = [], []
    for k in np.unique(key):
        m = key == k
        if m.sum() >= 3:
            means.append(float(z[m].mean()))
            vars_.append(float(z[m].var(ddof=1)))
    means, vars_ = np.asarray(means), np.asarray(vars_)
    if means.size < 2:
        raise ValueError("too few populated blocks; reduce `n_blocks`")

    # Intrinsic stationarity is about the INCREMENTS, not the levels:
    # E[Z(s+h) - Z(s)] = 0. A linear trend keeps the increment VARIANCE
    # stable while giving the increments a non-zero mean, so tying this
    # to variance drift would pass a trended field for the wrong reason.
    i2, j2 = np.triu_indices(z.size, k=1)
    lagvec = coords[j2] - coords[i2]
    dv = z[j2] - z[i2]
    # Orient every pair into the same half-space first. Binning on lag
    # DISTANCE alone averages the +x and -x pairs together, so a linear
    # trend cancels itself and passes -- the condition is about the lag
    # VECTOR, E[Z(s+h) - Z(s)] = 0 for each h.
    flip = lagvec[:, 0] < 0 if lagvec.shape[1] >= 1 else np.zeros(dv.size, bool)
    if lagvec.shape[1] >= 2:
        onaxis = np.isclose(lagvec[:, 0], 0.0)
        flip = np.where(onaxis, lagvec[:, 1] < 0, flip)
    dv = np.where(flip, -dv, dv)
    dd = np.linalg.norm(lagvec, axis=1)
    md = max_dist if max_dist is not None else (dd.max() / 2.0 if dd.size else 1.0)
    ke = np.clip(np.digitize(dd, np.linspace(0.0, md, n_bins + 1)) - 1,
                 0, n_bins - 1)
    inc_means = np.array([dv[ke == b].mean() if np.any(ke == b) else np.nan
                          for b in range(n_bins)])
    inc_sd = float(np.nanstd(dv)) or 1.0
    inc_bias = float(np.nanmax(np.abs(inc_means)) / inc_sd)

    overall_sd = float(z.std(ddof=1))
    mean_drift = float((means.max() - means.min()) / overall_sd) if overall_sd else 0.0
    vbar = float(vars_.mean())
    var_drift = float((vars_.max() - vars_.min()) / vbar) if vbar > 0 else 0.0

    mean_ok = mean_drift <= tol * 4.0
    var_ok = var_drift <= tol * 4.0
    second_order = bool(mean_ok and var_ok)
    return RichResult(
        title="Stationarity assessment",
        summary_lines=[("mean drift / sd", mean_drift),
                       ("variance drift", var_drift),
                       ("increment bias", inc_bias),
                       ("second-order plausible", second_order)],
        payload={"mean_stationary": bool(mean_ok),
                 "variance_stationary": bool(var_ok),
                 "second_order_plausible": second_order,
                 # increments can be stationary even when levels are not
                 "intrinsic_plausible": bool(inc_bias <= tol),
                 "increment_bias": inc_bias, "increment_means": inc_means,
                 "strict_if_gaussian": second_order,
                 "mean_drift": mean_drift, "variance_drift": var_drift,
                 "block_means": means, "block_vars": vars_,
                 "n_blocks_used": int(means.size), "tol": float(tol)},
    )


def cheatsheet():
    return "sprfss: strict > second-order > intrinsic; checks moment drift."
