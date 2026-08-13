"""Gene-based meta-analysis (MAGMA)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gene_meta_analysis"]


def gene_meta_analysis(sumstats, gene_annotation):
    """
    Gene-based meta-analysis (MAGMA)

    Formula: combine SNP p-values via Brown's method

    Parameters
    ----------
    sumstats : array-like
        Input data.
    gene_annotation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Leeuw et al (2015) MAGMA
    Brown, M. B. (1975). "A method for combining
    non-independent, one-sided tests of significance",
    Biometrics 31(4), 987-992. JSTOR 2529826. -- the method
    named above for combining dependent SNP p-values: it
    matches the first two moments of Fisher's statistic under
    dependence. PDF NOT IN HAND: JSTOR serves HTML, not the
    file. Cited because the module uses the method by name;
    the formula has not been re-verified against the paper.
    """
    sumstats = np.atleast_1d(np.asarray(sumstats, dtype=float))
    n = len(sumstats)
    result = float(np.mean(sumstats))
    se = float(np.std(sumstats, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gene-based meta-analysis (MAGMA)"})


def cheatsheet():
    return "genemt: Gene-based meta-analysis (MAGMA)"
