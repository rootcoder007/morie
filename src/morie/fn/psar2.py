"""Topological polar surface area (TPSA)."""

__all__ = ["polar_surface_area"]


def polar_surface_area(smiles):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Topological polar surface area (TPSA)

    Formula: sum tabulated PSA contributions per polar atom

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
    Ertl-Rohde-Selzer (2000)
    """
    raise NotImplementedError(
        "morie.fn.psar2.polar_surface_area is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "psar2: Topological polar surface area (TPSA)"
