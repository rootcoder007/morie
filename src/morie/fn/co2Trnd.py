"""Mauna Loa CO₂ trend (Keeling curve)."""

__all__ = ["co2_trend"]


def co2_trend(co2_monthly):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Mauna Loa CO₂ trend (Keeling curve)

    Formula: long-term + seasonal cycle decomposition

    Parameters
    ----------
    co2_monthly : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Keeling (1960)
    """
    raise NotImplementedError(
        "morie.fn.co2Trnd.co2_trend is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "co2Trnd: Mauna Loa CO₂ trend (Keeling curve)"


# compact alias per ledger/NAMING.md
co2trend = co2_trend
