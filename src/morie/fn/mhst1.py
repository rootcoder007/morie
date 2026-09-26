"""Mantel-Haenszel pooled OR."""

__all__ = ["mantel_haenszel_or"]


def mantel_haenszel_or(strata):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Mantel-Haenszel pooled OR

    Formula: OR_MH = sum(a_k d_k / n_k) / sum(b_k c_k / n_k)

    Parameters
    ----------
    strata : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mantel-Haenszel (1959)
    """
    raise NotImplementedError(
        "morie.fn.mhst1.mantel_haenszel_or is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "mhst1: Mantel-Haenszel pooled OR"
