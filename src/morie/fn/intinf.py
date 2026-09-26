"""Interaction information II(X;Y;Z)."""

__all__ = ["interaction_information"]


def interaction_information(pxyz):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Interaction information II(X;Y;Z)

    Formula: II = I(X;Y|Z) - I(X;Y)

    Parameters
    ----------
    pxyz : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McGill (1954)
    """
    raise NotImplementedError(
        "morie.fn.intinf.interaction_information is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "intinf: Interaction information II(X;Y;Z)"
