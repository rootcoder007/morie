"""Spatial Shannon-Wiener taxon diversity on a rectangular grid."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["frtaxd", "forest_taxon_diversity"]


def frtaxd(coords, species, grid=4):
    """
    Shannon-Wiener diversity H per grid cell and overall.

    For each cell (and for the pooled data),

        H = - sum_k p_k ln p_k,     p_k = n_k / n,

    over the taxa k present, with richness S (count of distinct taxa)
    and Pielou evenness J = H / ln S (J undefined, returned as NaN,
    when S = 1).

    Sources
    -------
    Shannon, C. E. (1948). A mathematical theory of communication.
    *Bell System Technical Journal*, 27, 379-423, Sec. 6, eq. for
    H = -K sum p_i log p_i (fetched-wave3/
    shannon-1948-mathematical-theory-of-communication.pdf); natural-log
    convention (K = 1, nats) as standard in ecology.
    Pielou, E. C. (1966). The measurement of diversity in different
    types of biological collections. *J. Theoretical Biology*, 13,
    131-144 (evenness J = H / ln S).
    Magurran, A. E. (2004). *Measuring Biological Diversity*, Blackwell
    (Shannon-Wiener usage in ecology; index form as above).

    Parameters
    ----------
    coords : array-like, (n, 2)
        Point locations of the individuals.
    species : array-like, (n,)
        Taxon labels (any hashable values).
    grid : int
        Number of grid cells per axis over the bounding box.

    Returns
    -------
    RichResult
        Keys: H (grid x grid matrix, NaN where empty), richness,
        counts (individuals per cell), H_overall, S_overall, J_overall.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if coords.shape[1] != 2:
        raise ValueError("`coords` must be (n, 2)")
    species = list(species)
    n = coords.shape[0]
    if len(species) != n:
        raise ValueError("`coords` and `species` must have equal length")
    g = int(grid)
    if g < 1:
        raise ValueError("`grid` must be a positive integer")

    def _shannon(labels):
        if not labels:
            return np.nan, 0, np.nan
        cnt = {}
        for s in labels:
            cnt[s] = cnt.get(s, 0) + 1
        tot = float(len(labels))
        h = -sum((c / tot) * np.log(c / tot) for c in cnt.values())
        s_rich = len(cnt)
        j = h / np.log(s_rich) if s_rich > 1 else np.nan
        return float(h), s_rich, j

    xmin, ymin = float(coords[:, 0].min()), float(coords[:, 1].min())
    xmax, ymax = float(coords[:, 0].max()), float(coords[:, 1].max())
    xr = xmax - xmin if xmax > xmin else 1.0
    yr = ymax - ymin if ymax > ymin else 1.0
    ix = np.clip((((coords[:, 0] - xmin) / xr) * g).astype(int), 0, g - 1)
    iy = np.clip((((coords[:, 1] - ymin) / yr) * g).astype(int), 0, g - 1)
    H = np.full((g, g), np.nan)
    richness = np.zeros((g, g), dtype=int)
    counts = np.zeros((g, g), dtype=int)
    for cx in range(g):
        for cy in range(g):
            labels = [species[i] for i in range(n) if int(ix[i]) == cx and int(iy[i]) == cy]
            h, s_rich, _ = _shannon(labels)
            H[cx, cy] = h
            richness[cx, cy] = s_rich
            counts[cx, cy] = len(labels)
    h_all, s_all, j_all = _shannon(species)
    return RichResult(payload={
        "H": H, "richness": richness, "counts": counts,
        "H_overall": h_all, "S_overall": int(s_all), "J_overall": j_all,
        "grid": g, "n": int(n),
        "method": "Shannon-Wiener diversity per grid cell (nats)",
    })


# long descriptive alias (stub-era name)
forest_taxon_diversity = frtaxd


def cheatsheet():
    return "frtaxd: Shannon-Wiener H = -sum p ln p per grid cell + Pielou J"
