"""Bayesian LASSO for genomic prediction."""

__all__ = ["bayes_lasso"]


def bayes_lasso(y, M, lam):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Bayesian LASSO for genomic prediction

    Formula: u_j ~ Laplace(0, lambda)

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Park-Casella (2008)
    """
    raise NotImplementedError(
        "morie.fn.bayslo.bayes_lasso is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayslo: Bayesian LASSO for genomic prediction"


# compact alias per ledger/NAMING.md
bayeslasso = bayes_lasso
