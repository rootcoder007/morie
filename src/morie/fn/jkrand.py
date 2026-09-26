"""Jackknife replicate weights variance."""

__all__ = ["jackknife_repl"]


def jackknife_repl(theta_replicates):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Jackknife replicate weights variance

    Formula: Var = (n-1)/n sum (theta_i - thetabar)^2

    Parameters
    ----------
    theta_replicates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wolter (2007)
    """
    raise NotImplementedError(
        "morie.fn.jkrand.jackknife_repl is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "jkrand: Jackknife replicate weights variance"


# compact alias per ledger/NAMING.md
jackkniferepl = jackknife_repl
