"""Years of life lost."""

__all__ = ["yll_calculation"]


def yll_calculation(deaths, ages, life_table):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Years of life lost

    Formula: YLL = N × (life_expectancy - age_at_death)

    Parameters
    ----------
    deaths : array-like
        Input data.
    ages : array-like
        Input data.
    life_table : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    WHO Global Burden of Disease
    """
    raise NotImplementedError(
        "morie.fn.yllyear.yll_calculation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "yllyear: Years of life lost"


# compact alias per ledger/NAMING.md
yllcalculation = yll_calculation
