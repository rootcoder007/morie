"""Crippen calculated LogP."""

__all__ = ["clogp_estimate"]


def clogp_estimate(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Crippen calculated LogP

    Formula: sum atomic contributions across 68 atom-types

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
    Wildman-Crippen (1999)
    """
    raise NotImplementedError(
        "morie.fn.clogp2.clogp_estimate is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "clogp2: Crippen calculated LogP"


# compact alias per ledger/NAMING.md
clogpestimate = clogp_estimate
