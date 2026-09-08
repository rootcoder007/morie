# morie.fn -- function file (rootcoder007/morie)
"""Selection coefficient from population differentiation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["selection_coefficient"]


def selection_coefficient(counts, n_total=None, generations=None,
                          n_e=None, alpha=0.05):
    r"""Detect selection by locus-specific departure from neutral F_ST.

    For allele counts across demes, the locus-specific differentiation

    .. math::
       F_{ST} = \frac{\sigma^2_p}{\bar p(1 - \bar p)}

    is compared with the genome-wide average. Foll and Gaggiotti's
    model decomposes each locus's coefficient into a
    population-specific term and a LOCUS-specific term
    :math:`\alpha_i`; a positive :math:`\alpha_i` indicates diversifying
    selection, a negative one balancing selection.

    The reason a single locus's :math:`F_{ST}` cannot be read on its own
    is that the neutral distribution is WIDE. Demographic history --
    bottlenecks, isolation by distance, unequal deme sizes -- inflates
    the variance of neutral :math:`F_{ST}` enormously, so outlier
    detection against a fixed threshold produces false positives in
    proportion to how structured the population is. The comparison here
    is against the EMPIRICAL genome-wide distribution, which absorbs
    that shared history.

    When ``generations`` and ``n_e`` are supplied, the per-generation
    selection coefficient follows from the diffusion relation
    :math:`s \approx \Delta p / (t\, p(1-p))`. That conversion assumes
    the allele frequency changed monotonically and deterministically;
    over few generations drift dominates and the number is noise.
    ``drift_dominates`` compares the observed change against the drift
    standard deviation :math:`\sqrt{p(1-p)t/(2N_e)}`.

    Parameters
    ----------
    counts : array-like, shape (n_loci, n_demes)
        Counts of the focal allele.
    n_total : array-like, optional
        Total alleles sampled per locus and deme. Assumed equal
        otherwise.
    generations : int, optional
    n_e : float, optional
        Effective population size.
    alpha : float

    Returns
    -------
    RichResult
        ``fst``, ``fst_mean``, ``alpha_locus``, ``outlier``,
        ``selection_type``, ``s``, ``drift_dominates``, ``q_value``.

    References
    ----------
    Foll and Gaggiotti (2008), *Genetics* 180:977-993.
    Weir and Cockerham (1984), *Evolution* 38:1358-1370.
    Lewontin and Krakauer (1973) for the outlier idea and its critique.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array([[10, 10], [20, 0], [15, 5]], float)
    >>> out = selection_coefficient(c, n_total=np.full((3, 2), 20.0))
    >>> int(np.argmax(out["fst"]))
    1
    """
    C = np.atleast_2d(np.asarray(counts, dtype=float))
    L, D = C.shape
    if D < 2:
        raise ValueError("need at least 2 demes, got %d." % D)
    if L < 2:
        raise ValueError(
            "need at least 2 loci: a single locus has no genome-wide "
            "distribution to be an outlier against."
        )
    N = (np.full_like(C, float(np.max(C)) * 2.0) if n_total is None
         else np.atleast_2d(np.asarray(n_total, dtype=float)))
    if N.shape != C.shape:
        raise ValueError("n_total must match counts in shape.")
    if np.any(C < 0) or np.any(C > N):
        raise ValueError("counts must lie between 0 and n_total.")

    p = C / np.maximum(N, 1e-12)
    pbar = p.mean(axis=1)
    var_p = p.var(axis=1, ddof=1) if D > 1 else np.zeros(L)
    denom = pbar * (1.0 - pbar)
    with np.errstate(divide="ignore", invalid="ignore"):
        fst = np.where(denom > 0, var_p / denom, np.nan)
    ok = np.isfinite(fst)
    fbar = float(np.mean(fst[ok])) if ok.any() else np.nan
    fsd = float(np.std(fst[ok], ddof=1)) if ok.sum() > 1 else np.nan

    # locus effect on the logit scale, as in the Foll-Gaggiotti split
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(np.clip(fst, 1e-9, 1 - 1e-9)
                    / (1 - np.clip(fst, 1e-9, 1 - 1e-9)))
        lbar = float(np.nanmean(lg))
        a_locus = lg - lbar

    z = (fst - fbar) / fsd if fsd and fsd > 0 else np.full(L, np.nan)
    import math
    pv = np.array([math.erfc(abs(v) / math.sqrt(2)) if np.isfinite(v) else np.nan
                   for v in z])
    # Benjamini-Hochberg, since every locus is tested
    q = np.full(L, np.nan)
    fin = np.isfinite(pv)
    if fin.any():
        idx = np.argsort(pv[fin])
        m = int(fin.sum())
        ranked = pv[fin][idx] * m / (np.arange(1, m + 1))
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        tmp = np.empty(m)
        tmp[idx] = np.clip(ranked, 0, 1)
        q[fin] = tmp
    outlier = np.where(np.isfinite(q), q < alpha, False)
    stype = np.where(~outlier, "neutral",
                     np.where(fst > fbar, "diversifying", "balancing"))

    s = drift = None
    if generations is not None and n_e is not None:
        t = float(generations)
        Ne = float(n_e)
        if t <= 0 or Ne <= 0:
            raise ValueError("generations and n_e must be positive.")
        dp = p[:, -1] - p[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(denom > 0, dp / (t * denom), np.nan)
        drift_sd = np.sqrt(np.maximum(denom, 0) * t / (2 * Ne))
        drift = np.abs(dp) < 2 * drift_sd
    return RichResult(
        payload={
            "estimate": fst,
            "fst": fst,
            "fst_mean": fbar,
            "fst_sd": fsd,
            "alpha_locus": a_locus,
            "alpha_note": (
                "locus effect on the logit scale; positive indicates "
                "diversifying selection, negative balancing"
            ),
            "z": z,
            "p_value": pv,
            "q_value": q,
            "outlier": outlier,
            "selection_type": stype,
            "n_outliers": int(np.sum(outlier)),
            "baseline_note": (
                "compared against the EMPIRICAL genome-wide distribution "
                "rather than a fixed threshold; demographic history inflates "
                "neutral F_ST variance, so a fixed cutoff yields false "
                "positives in proportion to how structured the population is"
            ),
            "s": s,
            "drift_dominates": drift,
            "drift_note": (
                None if drift is None else
                "the per-generation s assumes monotone deterministic change; "
                "where the observed shift is within two drift standard "
                "deviations sqrt(p(1-p)t/2Ne) the number is noise"
            ),
            "n_loci": int(L),
            "n_demes": int(D),
            "method": "Selection coefficient from F_ST outliers",
        }
    )


def cheatsheet():
    return (
        "sclvar: locus-wise F_ST against the genome-wide distribution with "
        "BH control and a drift check on the implied s"
    )
