"""Stationarity definitions: strict, second-order, intrinsic."""

from . import _array_core as np

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
    # All pairs i < j, streamed rather than materialised: n(n-1)/2 pairs
    # is 1.3 million at n = 1600, and building arrays of that size in
    # the pure-Python array core took minutes.
    import bisect
    import math as _m

    C = [[float(v) for v in row] for row in coords.tolist()]
    Z = [float(v) for v in z.tolist()]
    nz, dim = len(Z), len(C[0])
    dmax = 0.0
    for i in range(nz):
        ci = C[i]
        for j in range(i + 1, nz):
            cj = C[j]
            d2 = 0.0
            for t in range(dim):
                u = cj[t] - ci[t]
                d2 += u * u
            if d2 > dmax:
                dmax = d2
    dmax = _m.sqrt(dmax)
    md = max_dist if max_dist is not None else (dmax / 2.0 if nz > 1 else 1.0)
    edges = [float(v) for v in np.linspace(0.0, md, n_bins + 1).tolist()]
    bsum = [0.0] * n_bins
    bcnt = [0] * n_bins
    cnt, mean_, m2 = 0, 0.0, 0.0
    for i in range(nz):
        ci, zi = C[i], Z[i]
        for j in range(i + 1, nz):
            cj = C[j]
            d2 = 0.0
            for t in range(dim):
                u = cj[t] - ci[t]
                d2 += u * u
            dv = Z[j] - zi
            # orient every pair into the same half-space: the condition is
            # about the lag VECTOR, and binning on distance alone would let
            # the +h and -h pairs of a linear trend cancel
            lx = cj[0] - ci[0]
            if lx < 0 or (lx == 0.0 and dim >= 2 and cj[1] - ci[1] < 0):
                dv = -dv
            k = bisect.bisect_right(edges, _m.sqrt(d2)) - 1
            k = 0 if k < 0 else (n_bins - 1 if k > n_bins - 1 else k)
            bsum[k] += dv
            bcnt[k] += 1
            cnt += 1
            dlt = dv - mean_
            mean_ += dlt / cnt
            m2 += dlt * (dv - mean_)
    inc_means = [bsum[k] / bcnt[k] for k in range(n_bins) if bcnt[k]]
    # ddof=1 to match `overall_sd` below -- this is a scale normaliser for the
    # increments, and the two spreads in one result must be the same estimator.
    inc_sd = (_m.sqrt(m2 / (cnt - 1)) if cnt > 1 else 0.0) or 1.0
    inc_bias = float(max(abs(v) for v in inc_means) / inc_sd) if inc_means else float("nan")

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
