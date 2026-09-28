# morie.fn -- function file (rootcoder007/morie)
"""Wright's fixation index Fst from population allele frequencies."""

from . import _array_core as np
from ._containers import GenomicsResult


def fixation_index(
    allele_freqs: np.ndarray | list,
) -> GenomicsResult:
    """Wright's fixation index Fst from population allele frequencies.

    Fst = Var(p) / (p_bar * (1 - p_bar))

    where p_bar is the mean allele frequency across populations and
    Var(p) is the variance of allele frequencies across populations.
    When multiple loci are provided, Fst is averaged across loci. This is
    Wright's population-level Fst (populations weighted equally, no
    sample-size correction); it is not the Weir-Cockerham estimator, which
    needs sample sizes and heterozygosities.

    Parameters
    ----------
    allele_freqs : array-like, shape (n_pops, n_loci)
        Allele frequencies per population per locus. Each value in [0, 1].

    Returns
    -------
    GenomicsResult
        name="Fst", statistic=Fst estimate.

    Raises
    ------
    ValueError
        If fewer than 2 populations or frequencies outside [0, 1].

    References
    ----------
    Wright, S. (1951). The genetical structure of populations. Annals of
        Eugenics, 15(4), 323-354.
    """
    freqs = np.asarray(allele_freqs, dtype=float)
    if freqs.ndim == 1:
        freqs = freqs.reshape(-1, 1)
    if freqs.shape[0] < 2:
        raise ValueError("Need at least 2 populations.")
    if np.any(freqs < 0) or np.any(freqs > 1):
        raise ValueError("Allele frequencies must be in [0, 1].")

    p_bar = np.mean(freqs, axis=0)
    var_p = np.var(freqs, axis=0, ddof=0)

    denom = p_bar * (1.0 - p_bar)
    valid = denom > 0
    if not np.any(valid):
        return GenomicsResult(name="Fst", statistic=0.0, n=freqs.shape[0], extra={"n_loci": freqs.shape[1]})

    fst_per_locus = np.where(valid, var_p / denom, 0.0)
    fst = float(np.mean(fst_per_locus[valid]))

    return GenomicsResult(
        name="Fst",
        statistic=fst,
        n=freqs.shape[0],
        extra={"n_loci": freqs.shape[1], "per_locus": fst_per_locus.tolist()},
    )


fst = fixation_index


def cheatsheet() -> str:
    return "fixation_index({}) -> Fst fixation index (Weir-Cockerham estimator)."


# compact alias per ledger/NAMING.md
fixationindex = fixation_index
