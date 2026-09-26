"""Laplace transform."""

__all__ = ["laplace_transform"]


def laplace_transform(f, t, s):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Laplace transform

    Formula: L{f}(s) = ∫_0^∞ f(t) e^{-st} dt

    Parameters
    ----------
    f : array-like
        Input data.
    t : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    raise NotImplementedError(
        "morie.fn.laplT.laplace_transform is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "laplT: Laplace transform"
