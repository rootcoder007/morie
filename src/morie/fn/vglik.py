"""Variogram log-likelihood"""


def vario_loglik(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram log-likelihood

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vglik.vario_loglik is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_loglik


def cheatsheet() -> str:
    return "vario_loglik({}) -> Variogram log-likelihood"


# compact alias per ledger/NAMING.md
variologlik = vario_loglik
