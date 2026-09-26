"""Continuous wavelet transform (Morlet)."""

__all__ = ["cwt_morlet"]


def cwt_morlet(y, scales):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Continuous wavelet transform (Morlet)

    Formula: CWT(a,b) = (1/sqrt(a)) integral y(t) psi^*((t-b)/a) dt

    Parameters
    ----------
    y : array-like
        Input data.
    scales : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Torrence-Compo (1998)
    """
    raise NotImplementedError(
        "morie.fn.cwwsym.cwt_morlet is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "cwwsym: Continuous wavelet transform (Morlet)"


# compact alias per ledger/NAMING.md
cwtmorlet = cwt_morlet
