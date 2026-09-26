"""Next-generation matrix R0."""

__all__ = ["next_generation_matrix"]


def next_generation_matrix(FV_decomposition):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Next-generation matrix R0

    Formula: R0 = spectral radius of NGM

    Parameters
    ----------
    FV_decomposition : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diekmann-Heesterbeek-Roberts (2010)
    """
    raise NotImplementedError(
        "morie.fn.ngomtx.next_generation_matrix is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "ngomtx: Next-generation matrix R0"
