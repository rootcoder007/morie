"""Asymptotic expansion."""

__all__ = ["asymptotic_expansion"]


def asymptotic_expansion(f, x_inf):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Asymptotic expansion

    Formula: f ~ sum a_n φ_n as x->∞

    Parameters
    ----------
    f : array-like
        Input data.
    x_inf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Erdélyi (1956)
    """
    raise NotImplementedError(
        "morie.fn.asymp.asymptotic_expansion is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "asymp: Asymptotic expansion"
