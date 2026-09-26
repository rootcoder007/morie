"""CLR covariance matrix of a compositional sample."""

__all__ = ["aitchison_clr_covariance"]


def aitchison_clr_covariance(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CLR covariance matrix of a compositional sample

    Formula: Σ_clr = cov(clr(X))

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Sigma

    References
    ----------
    Pawlowsky-Glahn (2015)
    """
    raise NotImplementedError(
        "morie.fn.aitvarc.aitchison_clr_covariance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitvarc: CLR covariance matrix of a compositional sample"
