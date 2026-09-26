"""Asymmetric power ARCH."""

__all__ = ["aparch_dge"]


def aparch_dge(x, delta):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Asymmetric power ARCH

    Formula: sigma_t^delta = omega + alpha (|eps| - gamma eps)^delta + beta sigma_{t-1}^delta

    Parameters
    ----------
    x : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ding, Granger, Engle (1993)
    """
    raise NotImplementedError(
        "morie.fn.aparcm.aparch_dge is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aparcm: Asymmetric power ARCH"


# compact alias per ledger/NAMING.md
aparchdge = aparch_dge
