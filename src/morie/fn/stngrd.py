"""Spatial transformer network."""

__all__ = ["stn_spatial_transform"]


def stn_spatial_transform(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Spatial transformer network

    Formula: learned affine grid + bilinear sample

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jaderberg et al (2015)
    """
    raise NotImplementedError(
        "morie.fn.stngrd.stn_spatial_transform is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "stngrd: Spatial transformer network"
