"""CLR biplot loadings + scores from SVD."""

__all__ = ["aitchison_biplot"]


def aitchison_biplot(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CLR biplot loadings + scores from SVD

    Formula: U Σ V^T = clr(X) – cen

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: U, S, V

    References
    ----------
    Aitchison & Greenacre (2002)
    """
    raise NotImplementedError(
        "morie.fn.aitbi.aitchison_biplot is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitbi: CLR biplot loadings + scores from SVD"
