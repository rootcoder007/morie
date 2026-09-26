"""H-bond donor count."""

__all__ = ["hbond_donor_count"]


def hbond_donor_count(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    H-bond donor count

    Formula: count N-H + O-H groups

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
    Lipinski (1997)
    """
    raise NotImplementedError(
        "morie.fn.hbdon.hbond_donor_count is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hbdon: H-bond donor count"
