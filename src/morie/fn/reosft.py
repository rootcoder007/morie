"""REOS rapid-elimination of swill."""

__all__ = ["reos_filter"]


def reos_filter(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    REOS rapid-elimination of swill

    Formula: composite filter against PAINS-like nuisance compounds

    Parameters
    ----------
    smiles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Walters et al (1998)
    """
    raise NotImplementedError(
        "morie.fn.reosft.reos_filter is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "reosft: REOS rapid-elimination of swill"


# compact alias per ledger/NAMING.md
reosfilter = reos_filter
