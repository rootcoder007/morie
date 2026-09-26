"""Microsoft SR-CNN."""

__all__ = ["microsoft_sr"]


def microsoft_sr(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Microsoft SR-CNN

    Formula: spectral residual + CNN classifier

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
    Ren et al (2019)
    """
    raise NotImplementedError(
        "morie.fn.micrR.microsoft_sr is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "micrR: Microsoft SR-CNN"


# compact alias per ledger/NAMING.md
microsoftsr = microsoft_sr
