"""MCMC standard error of posterior mean."""

__all__ = ["mcmc_standard_error"]


def mcmc_standard_error(chains):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    MCMC standard error of posterior mean

    Formula: MCSE = sd / sqrt(ESS)

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
    Geyer (1992); Flegal et al. (2008)
    """
    raise NotImplementedError(
        "morie.fn.mcsec.mcmc_standard_error is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "mcsec: MCMC standard error of posterior mean"
