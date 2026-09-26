"""Separation of variables PDE."""

__all__ = ["pde_separation"]


def pde_separation(pde):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Separation of variables PDE

    Formula: u(x,t) = X(x)T(t)

    Parameters
    ----------
    pde : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    raise NotImplementedError(
        "morie.fn.pdesl.pde_separation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "pdesl: Separation of variables PDE"


# compact alias per ledger/NAMING.md
pdeseparation = pde_separation
