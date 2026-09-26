"""Fleiss kappa for multiple raters."""

__all__ = ["fleiss_kappa"]


def fleiss_kappa(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Fleiss kappa for multiple raters

    Formula: avg agreement adjusted for chance

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fleiss (1971)
    """
    raise NotImplementedError(
        "morie.fn.flskpa.fleiss_kappa is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "flskpa: Fleiss kappa for multiple raters"


# compact alias per ledger/NAMING.md
fleisskappa = fleiss_kappa
