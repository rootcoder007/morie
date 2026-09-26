"""Bayesian linear regression."""

__all__ = ["bayes_linear"]


def bayes_linear(y, X, prior_var):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Bayesian linear regression

    Formula: y = X beta + eps; beta ~ N(0, V); sigma ~ HalfCauchy

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    prior_var : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lindley-Smith (1972)
    """
    raise NotImplementedError(
        "morie.fn.bayreg.bayes_linear is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayreg: Bayesian linear regression"


# compact alias per ledger/NAMING.md
bayeslinear = bayes_linear
