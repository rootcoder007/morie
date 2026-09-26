"""Shor's quantum factoring (period-finding)."""

__all__ = ["shor_factoring"]


def shor_factoring(N):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Shor's quantum factoring (period-finding)

    Formula: QFT-based period of f(x)=a^x mod N

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shor (1994)
    """
    raise NotImplementedError(
        "morie.fn.shorE.shor_factoring is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "shorE: Shor's quantum factoring (period-finding)"


# compact alias per ledger/NAMING.md
shorfactoring = shor_factoring
