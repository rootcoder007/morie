"""Bayesian outlier detection."""

__all__ = ["bayes_outlier"]


def bayes_outlier(y, outlier_prior):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Bayesian outlier detection

    Formula: per-obs latent z indicating outlier component

    Parameters
    ----------
    y : array-like
        Input data.
    outlier_prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    West (1984)
    """
    raise NotImplementedError(
        "morie.fn.bayoutl.bayes_outlier is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayoutl: Bayesian outlier detection"


# compact alias per ledger/NAMING.md
bayesoutlier = bayes_outlier
