"""Fiedler (algebraic connectivity) eigenvalue λ_2 of L."""

__all__ = ["sgt_fiedler_value"]


def sgt_fiedler_value(A):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Fiedler (algebraic connectivity) eigenvalue λ_2 of L

    Formula: λ_2(L)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lam2

    References
    ----------
    Fiedler (1973)
    """
    raise NotImplementedError(
        "morie.fn.sgtfid.sgt_fiedler_value is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sgtfid: Fiedler (algebraic connectivity) eigenvalue λ_2 of L"
