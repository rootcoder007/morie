"""Fiedler vector -- eigenvector for λ_2."""

__all__ = ["sgt_fiedler_vector"]


def sgt_fiedler_vector(A):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Fiedler vector -- eigenvector for λ_2

    Formula: L v = λ_2 v

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: v

    References
    ----------
    Fiedler (1973)
    """
    raise NotImplementedError(
        "morie.fn.sgtfvc.sgt_fiedler_vector is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sgtfvc: Fiedler vector -- eigenvector for λ_2"
