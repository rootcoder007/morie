"""Newman modularity matrix B."""

__all__ = ["sgt_modularity_matrix"]


def sgt_modularity_matrix(A):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Newman modularity matrix B

    Formula: B_{ij} = A_{ij} - k_i k_j/(2m)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: B

    References
    ----------
    Newman (2006)
    """
    raise NotImplementedError(
        "morie.fn.sgtmodul.sgt_modularity_matrix is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sgtmodul: Newman modularity matrix B"
