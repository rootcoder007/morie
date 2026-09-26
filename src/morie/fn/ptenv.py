# morie.fn -- function file (rootcoder007/morie)
"""Point pattern Monte Carlo envelope"""


def pp_envelope(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Point pattern Monte Carlo envelope

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptenv.pp_envelope is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pp_e = pp_envelope


def cheatsheet() -> str:
    return "pp_envelope({}) -> Point pattern Monte Carlo envelope"


# compact alias per ledger/NAMING.md
ppenvelope = pp_envelope
