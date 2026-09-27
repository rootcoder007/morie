"""Gaussian kernel intensity (spatstat.explore::density.ppp at the points)."""

from morie.fn._rng import random_uniform
from morie.fn.kerint import kernel_intensity


def pattern():
    U = [float(u) for u in random_uniform(120, seed=21, stream=0)]
    return [[2 * U[2 * i], U[2 * i + 1]] for i in range(60)]


def test_matches_density_ppp():
    P = pattern()
    # density(X, sigma = 0.15, at = "points", edge = FALSE / TRUE / TRUE + diggle = TRUE)[1:4]
    ref = {
        "none": [15.27830275986, 15.81007650758, 12.74879430664, 24.78991934139],
        "uniform": [20.36689534952, 17.86577447956, 12.76588807138, 24.85439587018],
        "diggle": [16.8166678279, 18.26119067353, 13.3843427791, 26.4264746192],
    }
    for c, want in ref.items():
        got = kernel_intensity(P, (0, 2, 0, 1), 0.15, correction=c).value[:4]
        assert all(abs(a - b) < 1e-9 for a, b in zip(got, want))
    # densityfun(X, sigma = 0.15, edge = TRUE)(c(0.1, 1), c(0.1, 0.5))
    at = kernel_intensity(P, (0, 2, 0, 1), 0.15, at=[[0.1, 0.1], [1.0, 0.5]]).value
    assert abs(at[0] - 51.84394804586) < 1e-9 and abs(at[1] - 18.6163631496) < 1e-9


def test_kernel_mass_identity():
    # with no correction and no leave-one-out, integrating over the plane gives n
    P = [[0.5, 0.5]]
    r = kernel_intensity(P, (0, 1, 0, 1), 0.1, at=[[0.5, 0.5]], correction="none")
    assert abs(r.value[0] - 1 / (2 * 3.141592653589793 * 0.01)) < 1e-12
    assert abs(r.extra["edge"][0] - (1 - 2 * 2.866515718791939e-07) ** 2) < 1e-12
