"""Ames mutagenicity classification."""

__all__ = ["ames_mutagenicity"]


def ames_mutagenicity(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Ames mutagenicity classification

    Formula: RF classifier on structural alerts

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
    Hansen et al (2009); Honma et al (2019)
    """
    raise NotImplementedError(
        "morie.fn.ames3.ames_mutagenicity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "ames3: Ames mutagenicity classification"
