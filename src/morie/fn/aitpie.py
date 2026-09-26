"""Pielou evenness index."""

__all__ = ["compositional_pielou"]


def compositional_pielou(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Pielou evenness index

    Formula: J = H/log(D)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: J

    References
    ----------
    Pielou (1966)
    """
    raise NotImplementedError(
        "morie.fn.aitpie.compositional_pielou is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitpie: Pielou evenness index"
