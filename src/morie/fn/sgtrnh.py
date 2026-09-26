"""Randić connectivity index."""

__all__ = ["sgt_randic_index"]


def sgt_randic_index(A):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Randić connectivity index

    Formula: R = Σ_{ij ∈ E} (d_i d_j)^{-1/2}

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R

    References
    ----------
    Randić (1975)
    """
    raise NotImplementedError(
        "morie.fn.sgtrnh.sgt_randic_index is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sgtrnh: Randić connectivity index"


# compact alias per ledger/NAMING.md
sgtrandicindex = sgt_randic_index
