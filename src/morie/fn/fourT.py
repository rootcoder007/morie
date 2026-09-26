"""Fourier transform."""

__all__ = ["fourier_transform"]


def fourier_transform(f, x, k):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Fourier transform

    Formula: F(k) = ∫ f(x) e^{-2πikx} dx

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fourier (1822)
    """
    raise NotImplementedError(
        "morie.fn.fourT.fourier_transform is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "fourT: Fourier transform"
