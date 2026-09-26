"""RNA secondary structure (Zuker)."""

__all__ = ["rna_fold"]


def rna_fold(sequence):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    RNA secondary structure (Zuker)

    Formula: DP minimizing free energy

    Parameters
    ----------
    sequence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zuker (1989); Lorenz et al (2011) ViennaRNA
    """
    raise NotImplementedError(
        "morie.fn.rnafld.rna_fold is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "rnafld: RNA secondary structure (Zuker)"


# compact alias per ledger/NAMING.md
rnafold = rna_fold
