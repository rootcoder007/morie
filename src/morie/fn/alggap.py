"""Algebraic connectivity (lambda_2 of L)."""

__all__ = ["algebraic_connectivity"]


def algebraic_connectivity(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Algebraic connectivity (lambda_2 of L)

    Formula: second smallest eigenvalue of Laplacian

    Parameters
    ----------
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fiedler (1973)
    """
    raise NotImplementedError(
        "morie.fn.alggap.algebraic_connectivity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "alggap: Algebraic connectivity (lambda_2 of L)"
