import math

from morie.fn.mapproj import map_project
from morie.fn.projextra import (
    geoid_height,
    map_unproject,
    normal_gravity,
    robinson_project,
    rotated_grid,
    rotated_pole,
    web_mercator,
)


def test_inverses_round_trip():
    for lon, lat in ((10.0, 50.0), (-120.0, -30.0), (33.3, -71.2)):
        for proj in ("merc", "moll", "sinu"):
            x, y = map_project(lon, lat, proj).xy
            lo, la = map_unproject(x, y, proj)
            assert abs(lo - lon) < 1e-9 and abs(la - lat) < 1e-9
        lo, la = map_unproject(*web_mercator(lon, lat), "webmerc")
        assert abs(lo - lon) < 1e-9 and abs(la - lat) < 1e-9
        lo, la = map_unproject(*robinson_project(lon, lat), "robin")
        assert abs(lo - lon) < 1e-5 and abs(la - lat) < 1e-5  # PROJ's piecewise table is not exactly invertible


def test_robinson_equator_and_web_mercator_formula():
    x, y = robinson_project(90.0, 0.0)
    assert abs(x - 0.8487 * 6378137.0 * math.pi / 2) < 1e-6 and abs(y) < 1e-6
    x, y = web_mercator(10.0, 50.0)
    assert abs(y - 6378137.0 * math.log(math.tan(math.pi / 4 + math.radians(25.0)))) < 1e-6


def test_rotated_pole_geometry():
    # the rotated north pole sits at the true (pole_lon, pole_lat)
    lo, la = rotated_pole(0.0, 90.0, -162.0, 39.25, inverse=True)
    assert abs(la - 39.25) < 1e-9 and abs(lo + 162.0) < 1e-9
    for lon, lat in ((10.0, 50.0), (-120.0, -30.0)):
        r = rotated_pole(lon, lat, -162.0, 39.25)
        b = rotated_pole(r[0], r[1], -162.0, 39.25, inverse=True)
        assert abs(b[0] - lon) < 1e-9 and abs(b[1] - lat) < 1e-9
    g = rotated_grid([0.0], [90.0], -162.0, 39.25)
    assert abs(g.lat[0][0] - 39.25) < 1e-9


def test_geoid_single_harmonic():
    C = [[0.0], [0.0, 0.0], [0.0, 0.0, 2.43e-6]]
    S = [[0.0], [0.0, 0.0], [0.0, 0.0, -1.40e-6]]
    lon, lat = 30.0, 10.0
    p22 = math.sqrt(15.0) / 2 * math.cos(math.radians(lat)) ** 2  # fully normalised P22
    val = (2.43e-6 * math.cos(2 * math.radians(lon)) - 1.40e-6 * math.sin(2 * math.radians(lon))) * p22
    assert abs(geoid_height(lon, lat, C, S) - 3.986004418e14 / (6378137.0 * normal_gravity(lat)) * val) < 1e-12
    assert abs(normal_gravity(0.0) - 9.7803253359) < 1e-15
