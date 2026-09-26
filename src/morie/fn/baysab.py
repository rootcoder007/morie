"""BayesA prior on marker effects."""

__all__ = ["bayes_a_alpha"]


def bayes_a_alpha(y, M):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    BayesA prior on marker effects

    Formula: u_j ~ N(0, sigma_j^2); sigma_j^2 ~ scaled-inv-chi2

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Meuwissen-Hayes-Goddard (2001)
    """
    raise NotImplementedError(
        "morie.fn.baysab.bayes_a_alpha is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "baysab: BayesA prior on marker effects"


# compact alias per ledger/NAMING.md
bayesaalpha = bayes_a_alpha
