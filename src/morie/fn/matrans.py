"""Logit transform for proportion meta-analysis."""

__all__ = ["ma_logit_transform"]


def ma_logit_transform(p, n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Logit transform for proportion meta-analysis

    Formula: logit(p) = log(p/(1-p)); v = 1/(np) + 1/(n(1-p))

    Parameters
    ----------
    p : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: logit, var

    References
    ----------
    Nyaga et al. (2014)
    """
    raise NotImplementedError(
        "morie.fn.matrans.ma_logit_transform is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "matrans: Logit transform for proportion meta-analysis"
