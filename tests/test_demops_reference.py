"""demops against terra 1.9 (terrain, shade, flowAccumulation, watershed) and hand-derived cases.

terra values printed by R for DEM below (cell size 10); the fill, stream
order, D-infinity, MFD, curvature and viewshed cases are derived by hand.
tests/cross/test-morie_vs_terra_dem.R repeats the terra comparison live.
"""

import math

import pytest

from morie.fn.demops import (
    curvature,
    d8_flow_direction,
    dinf_flow_direction,
    fill_sinks,
    flow_accumulation,
    hillshade,
    mfd_flow_accumulation,
    stream_order,
    stream_power_index,
    terrain_indices,
    topographic_wetness_index,
    viewshed,
    watershed,
)

DEM = [
    [52.1, 51.4, 50.8, 50.9, 51.7, 52.6, 53.0],
    [50.7, 49.9, 49.1, 48.8, 49.6, 50.8, 51.9],
    [49.2, 48.1, 47.2, 46.9, 47.8, 49.3, 50.6],
    [48.8, 47.0, 45.9, 45.1, 46.2, 47.7, 49.4],
    [48.1, 46.3, 44.8, 43.6, 44.9, 46.5, 48.2],
    [47.9, 45.8, 44.1, 42.7, 44.0, 45.9, 47.6],
]
TERRA_slope = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    0.18080316624593346,
    0.18553300626566829,
    0.19379103730444933,
    0.21100353539394354,
    0.19209628460111164,
    float("nan"),
    float("nan"),
    0.17109137719797826,
    0.17417858217877047,
    0.1749690456656888,
    0.20378624168160095,
    0.20243981742976128,
    float("nan"),
    float("nan"),
    0.1632405704124609,
    0.15550833096646774,
    0.14853293341288493,
    0.19583374731695211,
    0.20441601672549053,
    float("nan"),
    float("nan"),
    0.17620709499469717,
    0.15681568534440049,
    0.10967174525555903,
    0.17857988704359293,
    0.19023633414482161,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_aspect = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    2.6810031390314446,
    2.8788452629871837,
    3.301505776748014,
    3.6339529961587265,
    3.7349427997465092,
    float("nan"),
    float("nan"),
    2.4792804498873506,
    2.7478944296297674,
    3.2834897081939571,
    3.7463722064029725,
    3.8839140330634812,
    float("nan"),
    float("nan"),
    2.1398538288190547,
    2.4805494847391092,
    3.2504048584070402,
    3.8645720069912848,
    3.9952606240440711,
    float("nan"),
    float("nan"),
    1.9369620964384495,
    2.1763409903998685,
    3.187015933011371,
    4.0744279811151793,
    4.196463466919405,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_TPI = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    0.074999999999995737,
    -0.14999999999999147,
    -0.45000000000000995,
    -0.25000000000000711,
    -0.012500000000002842,
    float("nan"),
    float("nan"),
    -0.37499999999999289,
    -0.39999999999999858,
    -0.5625,
    -0.25,
    0.049999999999997158,
    float("nan"),
    float("nan"),
    -0.29999999999999716,
    -0.22500000000000142,
    -0.81249999999999289,
    -0.27499999999999147,
    -0.16249999999999432,
    float("nan"),
    float("nan"),
    -0.25000000000000711,
    -0.26250000000000284,
    -1.1124999999999901,
    -0.3125,
    -0.23750000000000426,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_TRI = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    1.4249999999999989,
    1.4999999999999991,
    1.5750000000000011,
    1.6500000000000012,
    1.4875000000000007,
    float("nan"),
    float("nan"),
    1.4249999999999998,
    1.375,
    1.4375,
    1.5749999999999984,
    1.5,
    float("nan"),
    float("nan"),
    1.3000000000000016,
    1.2750000000000004,
    1.3125,
    1.5249999999999986,
    1.5374999999999988,
    float("nan"),
    float("nan"),
    1.4000000000000004,
    1.2624999999999993,
    1.3374999999999986,
    1.4125000000000005,
    1.4875000000000007,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_TRIriley = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    4.4944410108488428,
    4.6475800154488969,
    4.9719211578624236,
    5.2725705305856305,
    4.7780749261601194,
    float("nan"),
    float("nan"),
    4.4226688774991976,
    4.5276925690687051,
    4.6227697325304886,
    5.1146847410177658,
    5.0219518117958852,
    float("nan"),
    float("nan"),
    4.1327956639543686,
    4.0049968789001582,
    4.378355855797925,
    4.9658836071740504,
    5.0645829048402389,
    float("nan"),
    float("nan"),
    4.4766058571198784,
    3.9661064030103854,
    4.3185645763378337,
    4.5243784103454487,
    4.817675788178363,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_TRIrmsd = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    1.5890248582070692,
    1.6431676725154971,
    1.7578395831246956,
    1.864135188230726,
    1.6893045906526163,
    float("nan"),
    float("nan"),
    1.5636495771111893,
    1.600781059358211,
    1.6343959128681156,
    1.8083141320025113,
    1.7755280904564694,
    float("nan"),
    float("nan"),
    1.4611639196202468,
    1.41598022585063,
    1.5479825580412712,
    1.7557049866079424,
    1.7906004579469978,
    float("nan"),
    float("nan"),
    1.5827191791344415,
    1.4022303662380149,
    1.526843148460246,
    1.5996093273046397,
    1.7033056096895829,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_roughness = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    4.8999999999999986,
    4.5,
    4.8000000000000043,
    5.7000000000000028,
    5.2000000000000028,
    float("nan"),
    float("nan"),
    4.8000000000000043,
    4.7999999999999972,
    4.5,
    5.6999999999999957,
    5.6999999999999957,
    float("nan"),
    float("nan"),
    4.4000000000000057,
    4.5,
    4.1999999999999957,
    5.6999999999999957,
    5.7000000000000028,
    float("nan"),
    float("nan"),
    4.6999999999999957,
    4.2999999999999972,
    3.5,
    5,
    5.3999999999999986,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_flowdir = [
    2,
    2,
    4,
    4,
    4,
    8,
    8,
    2,
    2,
    4,
    4,
    8,
    8,
    8,
    2,
    2,
    2,
    4,
    8,
    8,
    8,
    1,
    2,
    2,
    4,
    8,
    8,
    8,
    1,
    2,
    2,
    4,
    8,
    8,
    16,
    1,
    1,
    1,
    0,
    16,
    16,
    16,
]
TERRA_acc = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    3,
    2,
    3,
    2,
    1,
    1,
    2,
    6,
    6,
    3,
    2,
    1,
    1,
    3,
    3,
    16,
    3,
    2,
    1,
    1,
    2,
    4,
    23,
    3,
    3,
    1,
    1,
    2,
    5,
    42,
    6,
    2,
    1,
]
TERRA_shade = [
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    0.66582656270012019,
    0.68743204276922221,
    0.7177786699046973,
    0.73066108998803014,
    0.71477313888318272,
    float("nan"),
    float("nan"),
    0.63921620273530722,
    0.66946461394668999,
    0.70436599257207888,
    0.72209588652204504,
    0.71399001220108227,
    float("nan"),
    float("nan"),
    0.59500210696221056,
    0.63412165591627889,
    0.68500325078154067,
    0.71099384863482817,
    0.70719397466765255,
    float("nan"),
    float("nan"),
    0.56715031860003717,
    0.59899122315313336,
    0.65568676927916969,
    0.68586520840829257,
    0.68111728943144323,
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
]
TERRA_ws = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


def _flat(g):
    return [v for row in g for v in row]


def _same(a, b, tol):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if y != y:
            assert x != x
        else:
            assert x == pytest.approx(y, abs=tol)


def test_terrain_indices_equal_terra():
    t = terrain_indices(DEM, res=10.0)
    for key, ref in [
        ("slope", TERRA_slope),
        ("aspect", TERRA_aspect),
        ("tpi", TERRA_TPI),
        ("tri", TERRA_TRI),
        ("tri_riley", TERRA_TRIriley),
        ("tri_rmsd", TERRA_TRIrmsd),
        ("roughness", TERRA_roughness),
    ]:
        _same(_flat(t[key]), ref, 1e-13)
    _same(_flat(hillshade(t.slope, t.aspect, angle=35, direction=200)), TERRA_shade, 1e-14)


def test_d8_accumulation_watershed_equal_terra():
    fd = d8_flow_direction(DEM, res=10.0)
    assert _flat(fd) == TERRA_flowdir
    assert _flat(flow_accumulation(fd)) == TERRA_acc
    assert _flat(watershed(fd, (5, 3))) == TERRA_ws


def test_fill_sinks_priority_flood():
    dem = [[5, 5, 5, 5], [5, 1, 2, 5], [5, 3, 4, 3], [5, 5, 5, 5]]
    f = fill_sinks(dem)
    # the pit drains through the edge cell of height 3 at (2, 3)
    assert f[1][1:3] == [3.0, 3.0] and f[2][1:3] == [3.0, 4.0]
    assert all(f[i][j] >= dem[i][j] for i in range(4) for j in range(4))
    assert fill_sinks(f) == f
    e = fill_sinks(dem, epsilon=0.01)
    # with epsilon every interior cell has a strictly lower neighbour
    for i in (1, 2):
        for j in (1, 2):
            nb = [e[i + a][j + b] for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
            assert min(nb) < e[i][j]


def test_stream_order_strahler_and_shreve():
    # two first-order tributaries join at (1, 1): Strahler 2, carried downstream; Shreve magnitude 2
    fd = [[2, 0, 8], [0, 4, 0], [0, 4, 0]]
    acc = flow_accumulation(fd)
    s = stream_order(fd, acc, 1)
    assert s[0][0] == 1 and s[0][2] == 1 and s[1][1] == 2 and s[2][1] == 2
    assert stream_order(fd, acc, 1, method="shreve")[2][1] == 2
    with pytest.raises(ValueError):
        stream_order(fd, acc, 1, method="horton")


def test_wetness_and_stream_power():
    a, b = 12.0, 0.3
    assert topographic_wetness_index([[a]], [[b]], res=5.0)[0][0] == pytest.approx(math.log(60.0 / math.tan(0.3)))
    assert stream_power_index([[a]], [[b]], res=5.0)[0][0] == pytest.approx(60.0 * math.tan(0.3))


def test_dinf_on_planes():
    # z decreasing eastward: direction 0; decreasing north-eastward (diagonal): pi/4; south: 3 pi/2
    east = [[3, 2, 1]] * 3
    assert dinf_flow_direction(east)[1][1] == pytest.approx(0.0, abs=1e-15)
    ne = [[2, 1, 0], [3, 2, 1], [4, 3, 2]]
    assert dinf_flow_direction(ne)[1][1] == pytest.approx(math.pi / 4, abs=1e-15)
    south = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    assert dinf_flow_direction(south)[1][1] == pytest.approx(3 * math.pi / 2, abs=1e-15)
    # a plane falling at angle theta in the first facet: r = atan(s2/s1)
    th = 0.3
    pl = [[-(x * math.cos(th) + (1 - y) * math.sin(th)) for x in range(3)] for y in range(3)]
    assert dinf_flow_direction(pl)[1][1] == pytest.approx(th, abs=1e-12)


def test_mfd_conserves_area_and_splits_by_slope():
    acc = mfd_flow_accumulation([[2, 1], [1, 0]])
    share = 1.0 / (2.0 + math.sqrt(2.0) ** 1.1)
    assert acc[1][0] == pytest.approx(1.0 + share, abs=1e-14)
    assert acc[1][1] == pytest.approx(4.0, abs=1e-14)


def test_curvature_zevenbergen_thorne():
    # z = x^2 + y^2 sampled at unit spacing: D = E = 1, total = -4; a plane has zero curvature
    bowl = [[(x - 1) ** 2 + (y - 1) ** 2 for x in range(3)] for y in range(3)]
    assert curvature(bowl).total[1][1] == pytest.approx(-4.0, abs=1e-15)
    plane = [[2 * x + 3 * y for x in range(3)] for y in range(3)]
    c = curvature(plane)
    assert (c.total[1][1], c.profile[1][1], c.plan[1][1]) == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)
    # z = x^2 on a slope z += y: profile and plan by the formulas
    s = [[(x - 1) ** 2 + (1 - y) for x in range(3)] for y in range(3)]
    D, E, F, G, H = 1.0, 0.0, 0.0, 0.0, 1.0
    assert curvature(s).profile[1][1] == pytest.approx(-2 * (D * G * G + E * H * H + F * G * H) / (G * G + H * H))
    assert curvature(s).plan[1][1] == pytest.approx(2 * (D * H * H + E * G * G - F * G * H) / (G * G + H * H))


def test_viewshed_line_of_sight():
    dem = [[0, 0, 0, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0]]
    v = viewshed(dem, (1, 0), observer_height=1.0)
    assert v[1][2] == 1 and v[1][3] == 0 and v[1][4] == 0
    assert v[0][4] == 1
    assert viewshed(dem, (1, 0), observer_height=20.0)[1][4] == 1


def test_validation():
    with pytest.raises(ValueError):
        terrain_indices([[1, 2], [3]])
    with pytest.raises(ValueError):
        terrain_indices([[1, 2, 3]] * 3, neighbors=6)
