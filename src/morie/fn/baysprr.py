"""Horseshoe sparsity prior."""

__all__ = ["sparsity_horseshoe"]


def sparsity_horseshoe(X, y, tau):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Horseshoe sparsity prior

    Formula: beta_j ~ N(0, tau lambda_j); lambda ~ C+(0,1)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Carvalho-Polson-Scott (2010)
    """
    raise NotImplementedError(
        "morie.fn.baysprr.sparsity_horseshoe is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "baysprr: Horseshoe sparsity prior"
