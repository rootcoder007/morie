"""CDCL conflict-driven clause learning."""

__all__ = ["cdcl"]


def cdcl(cnf):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CDCL conflict-driven clause learning

    Formula: DPLL + learned clauses + backjumping

    Parameters
    ----------
    cnf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Marques-Silva-Sakallah (1999)
    """
    raise NotImplementedError(
        "morie.fn.cdcl.cdcl is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "cdcl: CDCL conflict-driven clause learning"
