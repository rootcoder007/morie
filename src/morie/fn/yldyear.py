"""Years lived with disability."""

__all__ = ["yld_calculation"]


def yld_calculation(prevalence, disability, duration):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Years lived with disability

    Formula: YLD = prevalence × disability_weight × duration

    Parameters
    ----------
    prevalence : array-like
        Input data.
    disability : array-like
        Input data.
    duration : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    WHO GBD
    """
    raise NotImplementedError(
        "morie.fn.yldyear.yld_calculation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "yldyear: Years lived with disability"


# compact alias per ledger/NAMING.md
yldcalculation = yld_calculation
