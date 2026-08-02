# morie.fn -- function file (rootcoder007/morie)
"""CHOPIT anchoring-vignette model (King et al. 2004), simplified core."""

from . import _array_core as np
from scipy import optimize, stats

from ._richresult import RichResult

__all__ = ["chopit_vignette"]


def chopit_vignette(self_ratings, vignette_ratings, group=None, n_categories=None):
    r"""Vignette-anchored ordered probit for interpersonal comparability.

    King et al.'s insight: everyone rates the *same* hypothetical
    vignettes, so systematic differences in vignette ratings reveal
    differences in response thresholds (DIF), which can then be
    subtracted out of the self-assessments. This simplified core
    estimates, per group, an ordered-probit threshold vector from the
    vignette ratings (shared latent vignette means across groups),
    then reports each group's DIF shift and, as the corrected
    self-assessment, the group's latent mean fitted by ordered-probit
    ML under its own shifted thresholds -- directly comparable across
    groups because the DIF lives in the thresholds.

    Simplifications versus the full CHOPIT likelihood are documented,
    not hidden: thresholds shift by a single location parameter per
    group (no covariate model for each cut), vignette variances are
    pooled, and the latent location is anchored by mean vignette
    level = 0 (the likelihood identifies only tau - mu differences).

    Parameters
    ----------
    self_ratings : array-like, shape (n,)
        Ordinal self-assessments (1..K).
    vignette_ratings : array-like, shape (n, v)
        Ordinal ratings of the v common vignettes (NaN = missing).
    group : array-like, shape (n,), optional
        Group labels; default a single group.
    n_categories : int, optional
        K; default the max observed rating.

    Returns
    -------
    RichResult
        keys: ``dif_shift`` (dict group -> threshold shift, first
        group = 0), ``corrected_means`` (dict group -> DIF-corrected
        mean latent self-assessment), ``naive_means``, ``thresholds``
        (baseline cutpoints), ``vignette_means``, ``n``, ``method``.

    References
    ----------
    King, G., Murray, C. J. L., Salomon, J. A. & Tandon, A. (2004).
    Enhancing the validity and cross-cultural comparability of
    measurement in survey research. *APSR*, 98(1), 191-207.
    """
    y = np.asarray(self_ratings, dtype=float).ravel()
    Vg = np.asarray(vignette_ratings, dtype=float)
    if Vg.ndim != 2 or Vg.shape[0] != y.size:
        raise ValueError("vignette_ratings must be (n, v) matching self_ratings.")
    n, v = Vg.shape
    g = np.zeros(n, dtype=int) if group is None else np.unique(np.asarray(group).ravel(), return_inverse=True)[1]
    labels = ["g0"] if group is None else list(dict.fromkeys(np.asarray(group).ravel().tolist()))
    G = len(labels)
    K = int(n_categories) if n_categories is not None else int(max(np.nanmax(Vg), np.nanmax(y)))
    if K < 2:
        raise ValueError("need at least 2 response categories.")
    obs_ok = (y >= 1) & (y <= K)
    if not obs_ok.all():
        raise ValueError("self_ratings must lie in 1..n_categories.")

    # parameters: baseline thresholds (K-1, increasing via exp deltas),
    # group shifts (G-1), vignette means (v)
    def unpack(p):
        t0 = p[0]
        taus = np.concatenate([[t0], t0 + np.cumsum(np.exp(p[1 : K - 1]))]) if K > 2 else np.array([t0])
        shifts = np.concatenate([[0.0], p[K - 1 : K - 1 + G - 1]])
        mu = p[K - 1 + G - 1 :]
        return taus, shifts, mu

    def nll(p):
        taus, shifts, mu = unpack(p)
        total = 0.0
        for gi in range(G):
            rows = g == gi
            t = taus + shifts[gi]
            edges = np.concatenate([[-np.inf], t, [np.inf]])
            for j in range(v):
                col = Vg[rows, j]
                col = col[~np.isnan(col)].astype(int)
                if col.size == 0:
                    continue
                lo = stats.norm.cdf(edges[col - 1] - mu[j])
                hi = stats.norm.cdf(edges[col] - mu[j])
                total -= np.sum(np.log(np.clip(hi - lo, 1e-10, None)))
        return total

    p0 = np.concatenate([
        np.linspace(-1, 1, 1),          # t0
        np.log(np.full(max(K - 2, 0), 2.0 / max(K - 1, 1))),
        np.zeros(G - 1),
        np.nanmean(np.where(np.isnan(Vg), np.nan, Vg), axis=0) - (K + 1) / 2,
    ])
    res = optimize.minimize(nll, p0, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    taus, shifts, mu = unpack(res.x)
    # the likelihood only pins tau - mu differences; anchor the location by
    # setting the mean vignette level to zero (any joint shift is likelihood-
    # invariant, and the corrected means inherit this anchor consistently)
    c = float(mu.mean())
    taus = taus - c
    mu = mu - c

    # DIF-corrected self-assessment: per-group latent mean by ordered-probit
    # ML under that group's shifted thresholds. The shift lives in the
    # thresholds, so the fitted means are directly comparable across groups.
    naive = {}
    corrected = {}
    for gi, lab in enumerate(labels):
        rows = g == gi
        yg = y[rows].astype(int)
        naive[lab] = float(y[rows].mean())
        t = taus + shifts[gi]
        edges = np.concatenate([[-np.inf], t, [np.inf]])

        def nll_mean(m, yg=yg, edges=edges):
            lo = stats.norm.cdf(edges[yg - 1] - m)
            hi = stats.norm.cdf(edges[yg] - m)
            return -np.sum(np.log(np.clip(hi - lo, 1e-10, None)))

        r = optimize.minimize_scalar(nll_mean, bounds=(-5, 5), method="bounded")
        corrected[lab] = float(r.x)

    return RichResult(
        payload={
            "dif_shift": dict(zip(labels, shifts.tolist())),
            "corrected_means": corrected,
            "naive_means": naive,
            "thresholds": taus,
            "vignette_means": mu,
            "n": int(n),
            "method": "CHOPIT core: vignette-anchored thresholds, one DIF shift per group",
        }
    )


def cheatsheet():
    return "chopit: shared vignettes identify group threshold shifts; correct self-ratings"
