"""Molecular weight."""

__all__ = ["molecular_weight"]


def molecular_weight(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Molecular weight

    Formula: sum atomic weights of all atoms

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
    IUPAC atomic weights (2021)
    """
    raise NotImplementedError(
        "morie.fn.mwght.molecular_weight is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "mwght: Molecular weight"
