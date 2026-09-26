"""Build ILR contrast matrix V from a sequential binary partition."""

__all__ = ["aitchison_sbp_basis"]


def aitchison_sbp_basis(sign):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Build ILR contrast matrix V from a sequential binary partition

    Formula: V from SBP sign matrix per Egozcue & Pawlowsky-Glahn

    Parameters
    ----------
    sign : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V

    References
    ----------
    Egozcue (2005)
    """
    raise NotImplementedError(
        "morie.fn.aitsbp.aitchison_sbp_basis is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitsbp: Build ILR contrast matrix V from a sequential binary partition"
