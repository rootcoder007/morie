"""Spatial panel variance components."""


def sppvar(resid, unit_id):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial panel variance components.

    Category: SPanel

    Parameters
    ----------
    resid, unit_id : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.sppvar.sppvar is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sppvar_fn = sppvar


def cheatsheet() -> str:
    return "sppvar({}) -> Spatial panel variance components."
