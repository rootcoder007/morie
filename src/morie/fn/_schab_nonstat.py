# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-stationary covariance: parametric models and moving windows.

Schabenberger & Gotway (2005), Sec. 8.2.1 (pp. 422-423) and Sec. 8.3.1
(pp. 425-426).

Sec. 8.2.1 -- the point-source model of Hughes-Oliver et al. (1998a)
--------------------------------------------------------------------
  eq (8.1)  Corr[Z(s_i), Z(s_j)]
              = exp{-theta1 ||s_i - s_j|| exp{theta2 |c_i - c_j|
                                             + theta3 min[c_i, c_j]}}

with ``c_i = ||s_i - c||`` the distance from site i to the point source c.
Non-stationary because the correlation depends on where the pair sits
relative to the source, not only on their separation.

The book states three consequences, all of which this module reproduces and
:mod:`scripts.audit.schab_rest_verify` checks:

* ``theta2 = theta3 = 0`` reduces (8.1) to the exponential correlation model
  with practical range ``alpha = 3/theta1``.
* in general the pair behaves like an exponential model with practical range
  ``alpha(s_i,s_j) = 3 exp{-theta2|c_i-c_j| - theta3 min[c_i,c_j]} / theta1``.
* for two sites equidistant from the source, ``c_i = c_j``, that collapses to
  ``alpha = 3 exp{-theta3 ||s_i - c||} / theta1``.

Anisotropy (p. 423) replaces the distances by ``h*_ij = ||A(s_i - s_j)||``
and ``c*_i = ||A_c (s_i - c)||`` for anisotropy matrices A, A_c.

The book is explicit that ``theta1 > 0, theta2, theta3 >= 0`` are "necessary
but not sufficient" for the resulting matrix to be positive semi-definite,
and that when sufficient conditions cannot be derived "one must examine the
eigenvalues of the estimated covariance or correlation matrix to ensure that
at least the estimated model is valid". :func:`point_source_correlation`
therefore returns the minimum eigenvalue and a validity flag rather than
leaving the caller to assume validity.

The stub this module replaces printed ``C(s1,s2) = sigma(s1) sigma(s2)
rho(s1,s2)``, a generic heteroscedastic form that appears nowhere in
Sec. 8.2.1. The section's parametric model is (8.1).

Sec. 8.3.1 -- moving windows (Haas 1990, 1995)
-----------------------------------------------
Ordinary kriging restricted to a neighbourhood is *local kriging*:

    p_ok(Z; s0_i) = mu + c_i(theta)' Sigma_i(theta)^-1 (Z_i - 1 mu)

where the crucial detail is that "all n data points contribute to the
estimation of theta in local kriging" -- one global covariance model, only
the solve is local.

The moving-window method generalises this by re-estimating the semivariogram
*locally within the window*, so theta carries an index:

    p_ok(Z; s0_i) = mu + c_i(theta_i)' Sigma_i(theta_i)^-1 (Z_i - 1 mu)

That difference -- global theta versus per-window theta_i -- is the whole
content of the section, so :func:`moving_window_krige` takes a
``local_variogram`` switch and reports which was used.

Haas's window-size heuristic (p. 426) is implemented literally in
:func:`haas_window`: "enlarge a circle around the prediction site until at
least 35 sites are included, then include five sites at a time until there is
at least one pair of sites at each lag class and the nonlinear least squares
fit of the local semivariogram converges."

The book also names the cost of local neighbourhoods: the predictor "is no
longer best", and "as points are included and excluded in the neighborhoods
with changing prediction location, spurious discontinuities can be
introduced". Both are reported rather than hidden.
"""

import numpy as np

__all__ = [
    "exponential_correlation",
    "haas_window",
    "local_krige",
    "moving_window_krige",
    "point_source_correlation",
    "practical_range",
]


def _coords(s):
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if s.ndim != 2:
        raise ValueError(f"coordinates must be (n, d), got ndim={s.ndim}")
    if not np.all(np.isfinite(s)):
        raise ValueError("coordinates must be finite")
    return s


def _pairwise(a, b=None):
    a = _coords(a)
    b = a if b is None else _coords(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("coordinate sets have different dimensions")
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def exponential_correlation(h, theta1):
    """``exp(-theta1 h)``, the model (8.1) reduces to when theta2=theta3=0."""
    if theta1 <= 0:
        raise ValueError("theta1 must be positive")
    return np.exp(-float(theta1) * np.asarray(h, dtype=float))


def practical_range(theta1, theta2=0.0, theta3=0.0, ci=None, cj=None):
    """Practical range of the pair, p. 423.

    ``alpha(s_i,s_j) = 3 exp{-theta2 |c_i-c_j| - theta3 min[c_i,c_j]} /theta1``

    With ``theta2 = theta3 = 0`` this is ``3/theta1``, the practical range of
    the exponential model, i.e. the distance at which correlation falls to
    ``exp(-3)``, about 0.05.
    """
    if theta1 <= 0:
        raise ValueError("theta1 must be positive")
    if ci is None and cj is None:
        return 3.0 / float(theta1)
    ci = np.asarray(ci, dtype=float)
    cj = np.asarray(cj, dtype=float)
    return 3.0 * np.exp(-float(theta2) * np.abs(ci - cj)
                        - float(theta3) * np.minimum(ci, cj)) / float(theta1)


def point_source_correlation(coords, source, theta1, theta2=0.0, theta3=0.0,
                             anisotropy=None, source_anisotropy=None):
    """eq (8.1): the Hughes-Oliver point-source correlation matrix.

    Parameters
    ----------
    coords : (n, d) array
    source : (d,) array
        Location ``c`` of the point source.
    theta1 : float, > 0
    theta2, theta3 : float, >= 0
    anisotropy, source_anisotropy : (d, d) arrays, optional
        ``A`` and ``A_c`` of p. 423. When given, separations become
        ``h*_ij = ||A(s_i-s_j)||`` and ``c*_i = ||A_c(s_i - c)||``.

    Returns a dict with the correlation matrix, the source distances, the
    minimum eigenvalue and ``valid``. The book requires the eigenvalue check:
    the parameter constraints alone do not guarantee positive semi-definiteness.
    """
    s = _coords(coords)
    c = np.asarray(source, dtype=float).ravel()
    if c.size != s.shape[1]:
        raise ValueError(f"source has dimension {c.size} but coordinates have {s.shape[1]}")
    if theta1 <= 0:
        raise ValueError("theta1 must be positive (Sec. 8.2.1)")
    if theta2 < 0 or theta3 < 0:
        raise ValueError("theta2 and theta3 must be non-negative (Sec. 8.2.1)")

    diff = s[:, None, :] - s[None, :, :]
    if anisotropy is None:
        h = np.linalg.norm(diff, axis=-1)
    else:
        A = np.asarray(anisotropy, dtype=float)
        h = np.linalg.norm(diff @ A.T, axis=-1)

    dc = s - c
    if source_anisotropy is None:
        ci = np.linalg.norm(dc, axis=-1)
    else:
        Ac = np.asarray(source_anisotropy, dtype=float)
        ci = np.linalg.norm(dc @ Ac.T, axis=-1)

    inflate = np.exp(float(theta2) * np.abs(ci[:, None] - ci[None, :])
                     + float(theta3) * np.minimum(ci[:, None], ci[None, :]))
    corr = np.exp(-float(theta1) * h * inflate)
    np.fill_diagonal(corr, 1.0)

    eig = float(np.linalg.eigvalsh(0.5 * (corr + corr.T)).min())
    out = {
        "correlation": corr,
        "source_distance": ci,
        "separation": h,
        "min_eigenvalue": eig,
        "valid": bool(eig >= -1e-10),
        "theta": (float(theta1), float(theta2), float(theta3)),
    }
    if not out["valid"]:
        out["warning"] = (
            "the correlation matrix is not positive semi-definite (minimum "
            f"eigenvalue {eig:.3e}). Sec. 8.2.1 notes that theta1 > 0 and "
            "theta2, theta3 >= 0 are necessary but not sufficient, and that "
            "the eigenvalues must be examined to confirm the estimated model "
            "is valid")
    return out


def haas_window(coords, target, min_sites=35, step=5, lag_classes=None,
                max_sites=None):
    """Haas's window rule, p. 426.

    Enlarge a circle about ``target`` until at least ``min_sites`` sites are
    inside, then add ``step`` sites at a time until every lag class contains
    at least one pair. The book's third condition -- that the non-linear
    least squares fit of the local semivariogram converges -- is deferred to
    the caller, which sees ``converged`` from the fit; this function reports
    the geometric part and whether the lag-class condition was met.
    """
    s = _coords(coords)
    t = np.asarray(target, dtype=float).ravel()
    n = s.shape[0]
    if min_sites < 2:
        raise ValueError("min_sites must be at least 2")
    d = np.linalg.norm(s - t, axis=1)
    order = np.argsort(d, kind="stable")
    k = min(int(min_sites), n)
    cap = n if max_sites is None else min(n, int(max_sites))

    while True:
        idx = order[:k]
        radius = float(d[idx].max())
        pd = _pairwise(s[idx])
        iu = np.triu_indices(k, 1)
        h = pd[iu]
        nlag = int(lag_classes) if lag_classes else max(1, int(np.sqrt(h.size)))
        if h.size and h.max() > 0:
            edges = np.linspace(0.0, h.max(), nlag + 1)
            counts = np.histogram(h, bins=edges)[0]
            filled = bool(np.all(counts > 0))
        else:
            counts, filled = np.zeros(nlag, dtype=int), False
        if filled or k >= cap:
            return {
                "index": idx,
                "n_sites": int(k),
                "radius": radius,
                "lag_counts": counts,
                "all_lag_classes_filled": filled,
                "reached_cap": bool(k >= cap and not filled),
            }
        k = min(cap, k + int(step))


def _variogram_wls(h, gamma, counts, model="exponential"):
    """Fit a nugget-free exponential semivariogram by weighted least squares.

    ``gamma(h) = sill (1 - exp(-3 h / range))`` -- the parameterisation the
    book uses when it speaks of a practical range.
    """
    h = np.asarray(h, dtype=float)
    g = np.asarray(gamma, dtype=float)
    w = np.asarray(counts, dtype=float)
    ok = np.isfinite(g) & (w > 0) & (h > 0)
    if ok.sum() < 2:
        return {"sill": float("nan"), "range": float("nan"), "converged": False}
    h, g, w = h[ok], g[ok], w[ok]
    best, loss = None, np.inf
    for rng in np.geomspace(max(h.min(), 1e-9), h.max() * 3.0, 60):
        basis = 1.0 - np.exp(-3.0 * h / rng)
        denom = float((w * basis * basis).sum())
        if denom <= 0:
            continue
        sill = float((w * basis * g).sum() / denom)
        if sill <= 0:
            continue
        resid = g - sill * basis
        val = float((w * resid * resid).sum())
        if val < loss:
            loss, best = val, (sill, float(rng))
    if best is None:
        return {"sill": float("nan"), "range": float("nan"), "converged": False}
    return {"sill": best[0], "range": best[1], "converged": True, "wls": loss}


def _empirical_variogram(coords, z, n_lags=10):
    s = _coords(coords)
    z = np.asarray(z, dtype=float).ravel()
    pd = _pairwise(s)
    iu = np.triu_indices(s.shape[0], 1)
    h = pd[iu]
    sq = 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
    if h.size == 0 or h.max() <= 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(0.0, h.max(), int(n_lags) + 1)
    which = np.clip(np.digitize(h, edges) - 1, 0, int(n_lags) - 1)
    hbar = np.zeros(int(n_lags))
    gbar = np.full(int(n_lags), np.nan)
    cnt = np.zeros(int(n_lags))
    for b in range(int(n_lags)):
        m = which == b
        cnt[b] = m.sum()
        if cnt[b]:
            hbar[b] = h[m].mean()
            gbar[b] = sq[m].mean()
    return hbar, gbar, cnt


def _krige_at(coords, z, target, sill, rng, mu):
    s = _coords(coords)
    z = np.asarray(z, dtype=float).ravel()
    t = np.asarray(target, dtype=float).ravel()
    d = _pairwise(s)
    cov = sill * np.exp(-3.0 * d / rng)
    c0 = sill * np.exp(-3.0 * np.linalg.norm(s - t, axis=1) / rng)
    try:
        sol = np.linalg.solve(cov + 1e-10 * np.eye(s.shape[0]), z - mu)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(cov, z - mu, rcond=None)[0]
    return float(mu + c0 @ sol)


def local_krige(coords, z, targets, sill, rng, min_sites=35, step=5,
                local_mean=False):
    """Local ordinary kriging with a GLOBAL covariance model, p. 425.

    ``theta = (sill, range)`` is supplied once and used in every window;
    only the neighbourhood changes. This is the book's local kriging, as
    distinct from the moving-window method.
    """
    s = _coords(coords)
    z = np.asarray(z, dtype=float).ravel()
    tg = _coords(targets)
    preds, sizes = [], []
    for t in tg:
        win = haas_window(s, t, min_sites=min_sites, step=step)
        idx = win["index"]
        mu = float(z[idx].mean()) if local_mean else float(z.mean())
        preds.append(_krige_at(s[idx], z[idx], t, sill, rng, mu))
        sizes.append(win["n_sites"])
    return {
        "prediction": np.asarray(preds),
        "window_sizes": np.asarray(sizes),
        "theta_is_global": True,
        "sill": float(sill),
        "range": float(rng),
    }


def moving_window_krige(coords, z, targets, min_sites=35, step=5, n_lags=10,
                        local_mean=False, local_variogram=True):
    """Haas's moving-window kriging, p. 426.

    With ``local_variogram=True`` the semivariogram is re-estimated inside
    each window, so every prediction location carries its own
    ``theta_i = (sill_i, range_i)``. With it False the estimate is made once
    on all the data, which recovers local kriging -- the comparison the
    section is built around.

    The book's cautions are returned, not hidden: the local predictor "is no
    longer best", and neighbourhoods that change with prediction location can
    introduce "spurious discontinuities".
    """
    s = _coords(coords)
    z = np.asarray(z, dtype=float).ravel()
    tg = _coords(targets)
    if s.shape[0] != z.size:
        raise ValueError(f"{s.shape[0]} coordinates but {z.size} observations")

    gh, gg, gc = _empirical_variogram(s, z, n_lags)
    global_fit = _variogram_wls(gh, gg, gc)

    preds, sills, ranges, sizes, conv = [], [], [], [], []
    for t in tg:
        win = haas_window(s, t, min_sites=min_sites, step=step)
        idx = win["index"]
        if local_variogram:
            lh, lg, lc = _empirical_variogram(s[idx], z[idx], n_lags)
            fit = _variogram_wls(lh, lg, lc)
            if not fit["converged"]:
                fit = global_fit
        else:
            fit = global_fit
        mu = float(z[idx].mean()) if local_mean else float(z.mean())
        preds.append(_krige_at(s[idx], z[idx], t, fit["sill"], fit["range"], mu))
        sills.append(fit["sill"])
        ranges.append(fit["range"])
        sizes.append(win["n_sites"])
        conv.append(bool(fit["converged"]))

    return {
        "prediction": np.asarray(preds),
        "local_sill": np.asarray(sills),
        "local_range": np.asarray(ranges),
        "window_sizes": np.asarray(sizes),
        "converged": np.asarray(conv),
        "theta_is_global": not local_variogram,
        "global_sill": global_fit["sill"],
        "global_range": global_fit["range"],
        "caveats": (
            "a predictor that excludes observed sites is no longer best; "
            "windows that change with prediction location can introduce "
            "spurious discontinuities (Sec. 8.3.1)"),
    }
