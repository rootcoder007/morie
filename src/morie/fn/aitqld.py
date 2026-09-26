"""Aitchison interquartile distance."""

__all__ = ["compositional_quantile_dist"]


def compositional_quantile_dist(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Aitchison interquartile distance

    Formula: IQD = ||clr(Q3) - clr(Q1)||

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: iqd

    References
    ----------
    Pawlowsky-Glahn (2015)
    """
    raise NotImplementedError(
        "morie.fn.aitqld.compositional_quantile_dist is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitqld: Aitchison interquartile distance"
