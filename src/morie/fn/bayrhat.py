"""Potential scale reduction R-hat."""

__all__ = ["r_hat"]


def r_hat(chains):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Potential scale reduction R-hat

    Formula: sqrt(W + B/n) / W; should converge to 1

    Parameters
    ----------
    chains : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman-Rubin (1992); Vehtari et al (2021)
    """
    raise NotImplementedError(
        "morie.fn.bayrhat.r_hat is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayrhat: Potential scale reduction R-hat"


# compact alias per ledger/NAMING.md
rhat = r_hat
