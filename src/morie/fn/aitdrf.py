"""Method-of-moments fit of Dirichlet α."""

__all__ = ["dirichlet_fit_mom"]


def dirichlet_fit_mom(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Method-of-moments fit of Dirichlet α

    Formula: α̂_i = m_i s, s = (m_1(1-m_1)/v_1) - 1

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha

    References
    ----------
    Minka (2000)
    """
    raise NotImplementedError(
        "morie.fn.aitdrf.dirichlet_fit_mom is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitdrf: Method-of-moments fit of Dirichlet α"
