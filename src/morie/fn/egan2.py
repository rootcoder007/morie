"""Egan drug-like filter (PSA + LogP)."""

__all__ = ["egan_filter"]


def egan_filter(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Egan drug-like filter (PSA + LogP)

    Formula: PSA ≤132 Å², -1≤LogP≤5.88

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
    Egan-Merz-Baldwin (2000)
    """
    raise NotImplementedError(
        "morie.fn.egan2.egan_filter is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "egan2: Egan drug-like filter (PSA + LogP)"


# compact alias per ledger/NAMING.md
eganfilter = egan_filter
