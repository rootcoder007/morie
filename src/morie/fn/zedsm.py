"""Poisson disease mapping"""


def disease_map_pois(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Poisson disease mapping

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zedsm.disease_map_pois is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


dise = disease_map_pois


def cheatsheet() -> str:
    return "disease_map_pois({}) -> Poisson disease mapping"


# compact alias per ledger/NAMING.md
diseasemappois = disease_map_pois
