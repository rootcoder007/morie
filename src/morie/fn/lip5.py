"""Lipinski Rule of 5 oral-bioavailability filter."""

__all__ = ["lipinski_rule_of_5"]


def lipinski_rule_of_5(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Lipinski Rule of 5 oral-bioavailability filter

    Formula: MW≤500, LogP≤5, HBA≤10, HBD≤5; pass if ≥3 met

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
    Lipinski et al (1997, 2001)
    """
    raise NotImplementedError(
        "morie.fn.lip5.lipinski_rule_of_5 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "lip5: Lipinski Rule of 5 oral-bioavailability filter"
