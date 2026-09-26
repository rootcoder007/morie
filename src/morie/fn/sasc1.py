"""Synthetic accessibility score (SAscore)."""

__all__ = ["synthetic_accessibility"]


def synthetic_accessibility(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Synthetic accessibility score (SAscore)

    Formula: fragment frequency + complexity penalties; 1-10

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
    Ertl-Schuffenhauer (2009)
    """
    raise NotImplementedError(
        "morie.fn.sasc1.synthetic_accessibility is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sasc1: Synthetic accessibility score (SAscore)"
