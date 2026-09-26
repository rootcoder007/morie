"""Kinship from genotypes (KING-robust)."""

__all__ = ["kinship_estimator"]


def kinship_estimator(genotypes):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Kinship from genotypes (KING-robust)

    Formula: per-pair kinship via shared alleles

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
    Manichaikul et al (2010) KING
    """
    raise NotImplementedError(
        "morie.fn.kngshp.kinship_estimator is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "kngshp: Kinship from genotypes (KING-robust)"
