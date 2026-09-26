"""SELFIES encoding (robust to mutation)."""

__all__ = ["selfies_encode"]


def selfies_encode(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    SELFIES encoding (robust to mutation)

    Formula: context-free encoding guaranteeing valid molecule

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
    Krenn et al (2020) SELFIES
    """
    raise NotImplementedError(
        "morie.fn.selfgr.selfies_encode is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "selfgr: SELFIES encoding (robust to mutation)"


# compact alias per ledger/NAMING.md
selfiesencode = selfies_encode
