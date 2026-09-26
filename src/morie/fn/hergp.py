"""hERG cardiac potassium-channel inhibition risk."""

__all__ = ["herg_inhibition"]


def herg_inhibition(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    hERG cardiac potassium-channel inhibition risk

    Formula: DNN classifier on Morgan FP

    Parameters
    ----------
    smiles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cai et al (2019); Ogura et al (2019)
    """
    raise NotImplementedError(
        "morie.fn.hergp.herg_inhibition is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hergp: hERG cardiac potassium-channel inhibition risk"


# compact alias per ledger/NAMING.md
herginhibition = herg_inhibition
