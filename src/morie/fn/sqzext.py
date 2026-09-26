"""Squeeze-and-Excitation block."""

__all__ = ["squeeze_excite"]


def squeeze_excite(x, reduction):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Squeeze-and-Excitation block

    Formula: global pool + FC + sigmoid; multiply channels

    Parameters
    ----------
    x : array-like
        Input data.
    reduction : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hu-Shen-Sun (2018) SENet
    """
    raise NotImplementedError(
        "morie.fn.sqzext.squeeze_excite is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sqzext: Squeeze-and-Excitation block"


# compact alias per ledger/NAMING.md
squeezeexcite = squeeze_excite
