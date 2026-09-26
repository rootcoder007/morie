"""PSIS-LOO leave-one-out."""

__all__ = ["loo_psi"]


def loo_psi(log_lik):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    PSIS-LOO leave-one-out

    Formula: Pareto-smoothed importance sampling

    Parameters
    ----------
    log_lik : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vehtari-Gelman-Gabry (2017)
    """
    raise NotImplementedError(
        "morie.fn.bayloo.loo_psi is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayloo: PSIS-LOO leave-one-out"


# compact alias per ledger/NAMING.md
loopsi = loo_psi
