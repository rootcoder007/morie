"""Plasma protein binding fraction unbound."""

__all__ = ["plasma_protein_binding"]


def plasma_protein_binding(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Plasma protein binding fraction unbound

    Formula: regression on physchem descriptors

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
    Lambrinidis-Vallianatou-Tsantili-Kakoulidou (2015)
    """
    raise NotImplementedError(
        "morie.fn.ppbnd.plasma_protein_binding is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "ppbnd: Plasma protein binding fraction unbound"
