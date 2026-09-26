"""Functional scale (L²-norm)."""

__all__ = ["functional_scale"]


def functional_scale(f):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Functional scale (L²-norm)

    Formula: ||f|| = √∫f(t)² dt

    Parameters
    ----------
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    raise NotImplementedError(
        "morie.fn.fnscale.functional_scale is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "fnscale: Functional scale (L²-norm)"
