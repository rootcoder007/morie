"""GELU approximation via tanh."""

__all__ = ["gelu_tanh_approx"]


def gelu_tanh_approx(y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GELU approximation via tanh

    Formula: GELU(x) ~ 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hendrycks & Gimpel (2016)
    """
    raise NotImplementedError(
        "morie.fn.gelutn.gelu_tanh_approx is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "gelutn: GELU approximation via tanh"


# compact alias per ledger/NAMING.md
gelutanhapprox = gelu_tanh_approx
