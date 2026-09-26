"""Spatial wavelet MRA"""


def wavelet_mra_sp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial wavelet MRA

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxwmr.wavelet_mra_sp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


wave = wavelet_mra_sp


def cheatsheet() -> str:
    return "wavelet_mra_sp({}) -> Spatial wavelet MRA"


# compact alias per ledger/NAMING.md
waveletmrasp = wavelet_mra_sp
