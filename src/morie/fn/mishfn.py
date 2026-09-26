"""Mish activation."""

__all__ = ["mish_activation"]


def mish_activation(y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Mish activation

    Formula: Mish(x) = x * tanh(softplus(x))

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Misra (2019)
    """
    raise NotImplementedError(
        "morie.fn.mishfn.mish_activation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "mishfn: Mish activation"


# compact alias per ledger/NAMING.md
mishactivation = mish_activation
