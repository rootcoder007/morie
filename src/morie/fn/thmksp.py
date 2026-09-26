"""Thomas cluster process."""

__all__ = ["thomas_cluster"]


def thomas_cluster(lambda_p, mu, sigma):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Thomas cluster process

    Formula: Gaussian-distributed offspring around Poisson parents

    Parameters
    ----------
    lambda_p : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Thomas (1949)
    """
    raise NotImplementedError(
        "morie.fn.thmksp.thomas_cluster is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "thmksp: Thomas cluster process"


# compact alias per ledger/NAMING.md
thomascluster = thomas_cluster
