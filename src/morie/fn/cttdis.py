"""CTT item discrimination."""

__all__ = ["ctt_discrimination"]


def ctt_discrimination(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CTT item discrimination

    Formula: d_j = p_top27% - p_bottom27%

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kelley (1939)
    """
    raise NotImplementedError(
        "morie.fn.cttdis.ctt_discrimination is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "cttdis: CTT item discrimination"
