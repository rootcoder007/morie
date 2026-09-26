"""Walsh-Hadamard Transform with 1/sqrt(d) normalization (CRITICAL -- not 1/d)."""

__all__ = ["turboquant_walsh_hadamard_transform"]


def turboquant_walsh_hadamard_transform(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Walsh-Hadamard Transform with 1/sqrt(d) normalization (CRITICAL -- not 1/d)

    Formula: y = H x / sqrt(d);  H_{d x d} = Hadamard(d);  inverse: x = H y / sqrt(d)  (H^2 / d = I)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    TurboQuant MORIE integration -- morie/quant_ggml.c
    """
    raise NotImplementedError(
        "morie.fn.tqwht.turboquant_walsh_hadamard_transform is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "tqwht: Walsh-Hadamard Transform with 1/sqrt(d) normalization (CRITICAL -- not 1/d)"
