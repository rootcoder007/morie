"""Bias-correction spatial-disaggregation."""

__all__ = ["bcsd_downscaling"]


def bcsd_downscaling(gcm, obs):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Bias-correction spatial-disaggregation

    Formula: bias-correct -> spatial disaggregation via climatology

    Parameters
    ----------
    gcm : array-like
        Input data.
    obs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wood et al (2002)
    """
    raise NotImplementedError(
        "morie.fn.bcsd.bcsd_downscaling is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bcsd: Bias-correction spatial-disaggregation"
