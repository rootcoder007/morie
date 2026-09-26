"""Random non-Gaussian field"""


def random_nongauss(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Random non-Gaussian field

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsrng.random_nongauss is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


rand = random_nongauss


def cheatsheet() -> str:
    return "random_nongauss({}) -> Random non-Gaussian field"


# compact alias per ledger/NAMING.md
randomnongauss = random_nongauss
