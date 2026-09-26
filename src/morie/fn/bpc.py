"""Bits per character."""

__all__ = ["bits_per_character"]


def bits_per_character(log_probs, N):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Bits per character

    Formula: BPC = -(1/N) sum log_2 p(x_i)

    Parameters
    ----------
    log_probs : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hutter Prize benchmark
    """
    raise NotImplementedError(
        "morie.fn.bpc.bits_per_character is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bpc: Bits per character"
