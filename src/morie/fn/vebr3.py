"""Veber rule for oral bioavailability."""

__all__ = ["veber_rule"]


def veber_rule(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Veber rule for oral bioavailability

    Formula: rotatable bonds ≤10; PSA ≤140 Å²

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
    Veber et al (2002)
    """
    raise NotImplementedError(
        "morie.fn.vebr3.veber_rule is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "vebr3: Veber rule for oral bioavailability"


# compact alias per ledger/NAMING.md
veberrule = veber_rule
