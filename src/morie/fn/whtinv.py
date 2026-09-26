"""Inverse Walsh-Hadamard with 1/sqrt(d) normalization."""

__all__ = ["walsh_hadamard_inverse"]


def walsh_hadamard_inverse(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Inverse Walsh-Hadamard with 1/sqrt(d) normalization

    Formula: WHT^{-1}(y) = WHT(y) (because orthonormal with 1/sqrt(d))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hadamard (1893); Pratt (1969)
    """
    raise NotImplementedError(
        "morie.fn.whtinv.walsh_hadamard_inverse is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "whtinv: Inverse Walsh-Hadamard with 1/sqrt(d) normalization"
