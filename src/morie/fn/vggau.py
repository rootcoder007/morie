"""Gaussian variogram model"""


def vario_gaussian(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Gaussian variogram model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vggau.vario_gaussian is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_gaussian


def cheatsheet() -> str:
    return "vario_gaussian({}) -> Gaussian variogram model"


# compact alias per ledger/NAMING.md
variogaussian = vario_gaussian
