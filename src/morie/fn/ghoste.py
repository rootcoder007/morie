"""Ghose drug-like filter."""

__all__ = ["ghose_filter"]


def ghose_filter(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Ghose drug-like filter

    Formula: 160≤MW≤480, -0.4≤LogP≤5.6, 40≤MR≤130, 20≤atoms≤70

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
    Ghose-Viswanadhan-Wendoloski (1999)
    """
    raise NotImplementedError(
        "morie.fn.ghoste.ghose_filter is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "ghoste: Ghose drug-like filter"


# compact alias per ledger/NAMING.md
ghosefilter = ghose_filter
