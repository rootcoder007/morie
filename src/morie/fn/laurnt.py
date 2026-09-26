"""Laurent series (with negative powers)."""

__all__ = ["laurent_series"]


def laurent_series(f, c, order):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Laurent series (with negative powers)

    Formula: sum_{n=-∞}^∞ a_n (x-c)^n

    Parameters
    ----------
    f : array-like
        Input data.
    c : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Laurent (1843)
    """
    raise NotImplementedError(
        "morie.fn.laurnt.laurent_series is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "laurnt: Laurent series (with negative powers)"


# compact alias per ledger/NAMING.md
laurentseries = laurent_series
