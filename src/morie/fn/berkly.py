"""Berkeley Earth Kriging surface T."""

__all__ = ["berkeley_earth"]


def berkeley_earth(stations):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Berkeley Earth Kriging surface T

    Formula: weighted least squares + variogram

    Parameters
    ----------
    stations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rohde et al (2013)
    """
    raise NotImplementedError(
        "morie.fn.berkly.berkeley_earth is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "berkly: Berkeley Earth Kriging surface T"


# compact alias per ledger/NAMING.md
berkeleyearth = berkeley_earth
