"""BayesC pi."""

__all__ = ["bayes_c_pi"]


def bayes_c_pi(y, M, pi):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    BayesC pi

    Formula: u_j ~ N(0, sigma_b^2) with prob (1-pi); 0 otherwise

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Habier et al (2011)
    """
    raise NotImplementedError(
        "morie.fn.bayscm.bayes_c_pi is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayscm: BayesC pi"


# compact alias per ledger/NAMING.md
bayescpi = bayes_c_pi
