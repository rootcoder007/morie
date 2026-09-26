"""Blood-brain barrier permeability classifier."""

__all__ = ["bbb_permeability"]


def bbb_permeability(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Blood-brain barrier permeability classifier

    Formula: random forest on physchem + fingerprint features

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
    Martins et al (2012); Vilar et al (2010)
    """
    raise NotImplementedError(
        "morie.fn.bbbpr.bbb_permeability is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bbbpr: Blood-brain barrier permeability classifier"
