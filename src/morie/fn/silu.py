"""SiLU / Swish activation."""

__all__ = ["silu_swish"]


def silu_swish(y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    SiLU / Swish activation

    Formula: SiLU(x) = x * sigmoid(x)

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
    Elfwing, Uchibe, Doya (2018); Ramachandran et al. (2017) Swish
    """
    raise NotImplementedError(
        "morie.fn.silu.silu_swish is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "silu: SiLU / Swish activation"


# compact alias per ledger/NAMING.md
siluswish = silu_swish
