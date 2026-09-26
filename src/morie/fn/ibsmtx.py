"""Identity-by-state matrix."""

__all__ = ["ibs_matrix"]


def ibs_matrix(genotypes):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Identity-by-state matrix

    Formula: per-pair shared allele count / total

    Parameters
    ----------
    genotypes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Purcell et al (2007) PLINK
    """
    raise NotImplementedError(
        "morie.fn.ibsmtx.ibs_matrix is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "ibsmtx: Identity-by-state matrix"


# compact alias per ledger/NAMING.md
ibsmatrix = ibs_matrix
