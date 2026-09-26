"""Tent and circular semivariograms against gstat::variogramLine."""

import math

from morie.fn.spcirc import schabenberger_circular_variogram
from morie.fn.sptent import schabenberger_tent_variogram

H = [0.0, 0.3, 1.0, 1.7, 2.5, 3.0, 4.0]


def test_circular_matches_gstat_cir():
    # gstat::variogramLine(vgm(psill = 2, "Cir", range = 3, nugget = 0.5), dist_vector = H)
    ref = [0.0, 0.75422285686090684, 1.33283437683622985, 1.86154890899851444, 2.34079003836406629, 2.5, 2.5]
    g = schabenberger_circular_variogram(H, nugget=0.5, sill=2.0, range=3.0)["gamma"]
    for a, b in zip(g, ref):
        assert abs(a - b) < 1e-12


def test_tent_matches_gstat_lin():
    # gstat::variogramLine(vgm(psill = 2, "Lin", range = 3, nugget = 0.5), dist_vector = H)
    ref = [0.0, 0.69999999999999996, 1.16666666666666652, 1.63333333333333330, 2.16666666666666696, 2.5, 2.5]
    g = schabenberger_tent_variogram(H, nugget=0.5, sill=2.0, range=3.0)["gamma"]
    for a, b in zip(g, ref):
        assert abs(a - b) < 1e-12


def test_circular_is_the_printed_formula():
    # Schabenberger & Gotway (2005) p. 146: R2(h) = (2/pi){acos(h/a) - (h/a) sqrt(1 - h^2/a^2)}
    a = 2.0
    for h in (0.2, 0.9, 1.6):
        u = h / a
        r = 2.0 / math.pi * (math.acos(u) - u * math.sqrt(1.0 - u * u))
        g = schabenberger_circular_variogram([h], sill=1.0, range=a)["gamma"][0]
        assert abs(g - (1.0 - r)) < 1e-15
