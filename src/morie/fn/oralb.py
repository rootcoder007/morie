"""Oral bioavailability fraction."""

__all__ = ["oral_bioavailability"]


def oral_bioavailability(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Oral bioavailability fraction

    Formula: composite filter (Lipinski + Veber + Egan); regression refinement

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
    Veber et al (2002); Hou-Xu (2003)
    """
    raise NotImplementedError(
        "morie.fn.oralb.oral_bioavailability is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "oralb: Oral bioavailability fraction"
