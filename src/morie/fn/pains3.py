"""Pan-assay interference compound filter (PAINS)."""

__all__ = ["pains_filter"]


def pains_filter(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Pan-assay interference compound filter (PAINS)

    Formula: 480 SMARTS patterns flagging promiscuous binders

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
    Baell-Holloway (2010)
    """
    raise NotImplementedError(
        "morie.fn.pains3.pains_filter is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "pains3: Pan-assay interference compound filter (PAINS)"


# compact alias per ledger/NAMING.md
painsfilter = pains_filter
