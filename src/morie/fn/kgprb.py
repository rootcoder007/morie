# morie.fn -- function file (rootcoder007/morie)
"""Kriging probability map"""


def kriging_prob_map(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging probability map

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgprb.kriging_prob_map is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_prob_map


def cheatsheet() -> str:
    return "kriging_prob_map({}) -> Kriging probability map"


# compact alias per ledger/NAMING.md
krigingprobmap = kriging_prob_map
