"""Deterministic seeded RNG factory for reproducible bootstrap."""

__all__ = ["boot_rng_seeded"]


def boot_rng_seeded(seed):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Deterministic seeded RNG factory for reproducible bootstrap

    Formula: rng = numpy.random.default_rng(seed)

    Parameters
    ----------
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rng

    References
    ----------
    PCG64 (O'Neill 2014)
    """
    raise NotImplementedError(
        "morie.fn.btrnd.boot_rng_seeded is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "btrnd: Deterministic seeded RNG factory for reproducible bootstrap"


# compact alias per ledger/NAMING.md
bootrngseeded = boot_rng_seeded
