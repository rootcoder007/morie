"""Small-world coefficient sigma (Humphries-Gurney)."""

__all__ = ["small_world_sigma"]


def small_world_sigma(y, A):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Small-world coefficient sigma (Humphries-Gurney)

    Formula: sigma = (C / C_rand) / (L / L_rand)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Humphries & Gurney (2008)
    """
    raise NotImplementedError(
        "morie.fn.smallw.small_world_sigma is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "smallw: Small-world coefficient sigma (Humphries-Gurney)"
