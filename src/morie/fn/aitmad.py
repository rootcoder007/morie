"""MAD of CLR-transformed compositions."""

__all__ = ["compositional_mad"]


def compositional_mad(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    MAD of CLR-transformed compositions

    Formula: MAD = median(|z - median(z)|)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mad

    References
    ----------
    Filzmoser et al. (2018)
    """
    raise NotImplementedError(
        "morie.fn.aitmad.compositional_mad is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitmad: MAD of CLR-transformed compositions"
