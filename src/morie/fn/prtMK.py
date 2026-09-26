"""Pre-whitened Mann-Kendall."""

__all__ = ["prewhitening_mk"]


def prewhitening_mk(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Pre-whitened Mann-Kendall

    Formula: remove AR(1) before MK

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yue-Pilon-Cavadias (2002)
    """
    raise NotImplementedError(
        "morie.fn.prtMK.prewhitening_mk is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "prtMK: Pre-whitened Mann-Kendall"


# compact alias per ledger/NAMING.md
prewhiteningmk = prewhitening_mk
