"""Linear utility function for spatial voting."""

from . import _array_core as np

from ._containers import SpatialResult


def svlut(ideal, pos):
    """Linear utility function for spatial voting.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    stat = -float(np.linalg.norm(ideal - pos))
    return SpatialResult(
        name="Linear Spatial Utility",
        statistic=float(stat),
        extra={"euclidean_distance": -stat},
    )


svlut = svlut  # alias


def cheatsheet() -> str:
    return "svlut({}) -> Linear utility function for spatial voting."
