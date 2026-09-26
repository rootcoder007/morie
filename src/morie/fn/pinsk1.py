"""Pinsker's inequality (TV vs KL)."""

__all__ = ["pinsker_inequality"]


def pinsker_inequality(p, q):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Pinsker's inequality (TV vs KL)

    Formula: ||p-q||_TV <= sqrt(0.5 D_KL(p||q))

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pinsker (1964)
    """
    raise NotImplementedError(
        "morie.fn.pinsk1.pinsker_inequality is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "pinsk1: Pinsker's inequality (TV vs KL)"
