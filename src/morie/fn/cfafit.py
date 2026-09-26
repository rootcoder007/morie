"""CFA fit indices (CFI, RMSEA, SRMR, TLI)."""

__all__ = ["cfa_fit_indices"]


def cfa_fit_indices(fit):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CFA fit indices (CFI, RMSEA, SRMR, TLI)

    Formula: chi-sq based + residual based

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
    Hu-Bentler (1999)
    """
    raise NotImplementedError(
        "morie.fn.cfafit.cfa_fit_indices is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "cfafit: CFA fit indices (CFI, RMSEA, SRMR, TLI)"


# compact alias per ledger/NAMING.md
cfafitindices = cfa_fit_indices
