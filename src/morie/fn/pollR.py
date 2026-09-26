"""Pollard's rho factoring."""

__all__ = ["pollards_rho"]


def pollards_rho(n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Pollard's rho factoring

    Formula: cycle-detect on x²+c mod n

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pollard (1975)
    """
    raise NotImplementedError(
        "morie.fn.pollR.pollards_rho is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "pollR: Pollard's rho factoring"


# compact alias per ledger/NAMING.md
pollardsrho = pollards_rho
