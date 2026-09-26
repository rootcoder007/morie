"""Gelman-Rubin R-hat convergence diagnostic."""

__all__ = ["r_hat_convergence"]


def r_hat_convergence(chains):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Gelman-Rubin R-hat convergence diagnostic

    Formula: R_hat = sqrt( (n-1)/n + (1/n) B/W )

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
    Gelman & Rubin (1992); Vehtari, Gelman, Simpson, Carpenter, Burkner (2021) split-R-hat
    """
    raise NotImplementedError(
        "morie.fn.rhatc.r_hat_convergence is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "rhatc: Gelman-Rubin R-hat convergence diagnostic"
