"""Swin shifted-window attention."""

__all__ = ["swin_transformer"]


def swin_transformer(x, window_size):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Swin shifted-window attention

    Formula: window MSA + cyclic shift

    Parameters
    ----------
    x : array-like
        Input data.
    window_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu et al (2021) Swin
    """
    raise NotImplementedError(
        "morie.fn.swintr.swin_transformer is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "swintr: Swin shifted-window attention"
