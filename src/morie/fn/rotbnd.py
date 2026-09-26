"""Rotatable bond count."""

__all__ = ["rotatable_bond_count"]


def rotatable_bond_count(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Rotatable bond count

    Formula: non-ring single bonds excluding terminal

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
    Veber (2002)
    """
    raise NotImplementedError(
        "morie.fn.rotbnd.rotatable_bond_count is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "rotbnd: Rotatable bond count"
