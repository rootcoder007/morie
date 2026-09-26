"""North Atlantic Oscillation."""

__all__ = ["nao_index"]


def nao_index(slp):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    North Atlantic Oscillation

    Formula: normalized SLP difference Azores − Iceland

    Parameters
    ----------
    slp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hurrell (1995)
    """
    raise NotImplementedError(
        "morie.fn.naoIdx.nao_index is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "naoIdx: North Atlantic Oscillation"


# compact alias per ledger/NAMING.md
naoindex = nao_index
