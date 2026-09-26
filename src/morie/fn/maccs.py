"""MACCS 166-bit structural keys."""

__all__ = ["maccs_keys"]


def maccs_keys(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    MACCS 166-bit structural keys

    Formula: 166 hand-crafted SMARTS substructure indicators

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
    Durant et al (2002) JCIM
    """
    raise NotImplementedError(
        "morie.fn.maccs.maccs_keys is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "maccs: MACCS 166-bit structural keys"


# compact alias per ledger/NAMING.md
maccskeys = maccs_keys
