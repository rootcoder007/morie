"""Spatial wavelet analysis"""


def wavelet_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial wavelet analysis

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxwlt.wavelet_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


wave = wavelet_spatial


def cheatsheet() -> str:
    return "wavelet_spatial({}) -> Spatial wavelet analysis"


# compact alias per ledger/NAMING.md
waveletspatial = wavelet_spatial
