"""Bayesian shrinkage (horseshoe / Laplace)."""

__all__ = ["shrinkage_bayes"]


def shrinkage_bayes(X, y, prior_family):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Bayesian shrinkage (horseshoe / Laplace)

    Formula: beta_j ~ N(0, lambda_j tau); lambda_j ~ C+(0,1)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    prior_family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Carvalho-Polson-Scott (2010) horseshoe
    """
    raise NotImplementedError(
        "morie.fn.shrinkbm.shrinkage_bayes is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "shrinkbm: Bayesian shrinkage (horseshoe / Laplace)"


# compact alias per ledger/NAMING.md
shrinkagebayes = shrinkage_bayes
