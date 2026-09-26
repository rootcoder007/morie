"""Space deformation non-stationary covariance"""


def space_deformation(coords=None, data=None, *, n=50, dims=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Space deformation model for non-stationary covariance.

    Transforms spatial coordinates via a smooth mapping f: R^d -> R^d
    such that the process is stationary in the deformed space.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.sgspd.space_deformation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "sgspd"
alias = "space_deformation"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
space_deformation = space_deformation


def cheatsheet() -> str:
    return "space_deformation({}) -> Space deformation non-stationary covariance"
