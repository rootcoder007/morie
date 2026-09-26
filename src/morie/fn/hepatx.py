"""Drug-induced liver injury (DILI) classification."""

__all__ = ["hepatotoxicity"]


def hepatotoxicity(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Drug-induced liver injury (DILI) classification

    Formula: ensemble classifier on FP + pharmacophore features

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
    Chen et al (2016) DILI
    """
    raise NotImplementedError(
        "morie.fn.hepatx.hepatotoxicity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hepatx: Drug-induced liver injury (DILI) classification"
