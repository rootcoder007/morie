"""H-bond acceptor count."""

__all__ = ["hbond_acceptor_count"]


def hbond_acceptor_count(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    H-bond acceptor count

    Formula: count N + O atoms (Lipinski definition)

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
        "morie.fn.hbacc.hbond_acceptor_count is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hbacc: H-bond acceptor count"
