"""Estrada index -- sum exp eigenvalues."""

__all__ = ["sgt_estrada_index"]


def sgt_estrada_index(A):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Estrada index -- sum exp eigenvalues

    Formula: EE(G) = Σ_i exp(λ_i)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: EE

    References
    ----------
    Estrada (2000)
    """
    raise NotImplementedError(
        "morie.fn.sgtest.sgt_estrada_index is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sgtest: Estrada index -- sum exp eigenvalues"
