"""Satorra-Bentler chi-square correction."""

__all__ = ["sem_sb_chi_sq"]


def sem_sb_chi_sq(fit):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Satorra-Bentler chi-square correction

    Formula: chi-sq_SB = chi-sq_ML / scaling correction

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Satorra-Bentler (1994)
    """
    raise NotImplementedError(
        "morie.fn.semsbn.sem_sb_chi_sq is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "semsbn: Satorra-Bentler chi-square correction"


# compact alias per ledger/NAMING.md
semsbchisq = sem_sb_chi_sq
