"""rastops: focal statistics, kernels, morphology and raster operations recomputed by hand (terra checked in tests/cross)."""

import math

import pytest

from morie.fn.rastops import (
    canny_edges,
    cost_distance,
    distance_transform,
    filter_kernel,
    focal_filter,
    focal_statistics,
    grey_morphology,
    raster_aggregate,
    raster_disaggregate,
    raster_mask,
    raster_resample,
    zonal_statistics,
)

G = [[1.0, 5.0, 2.0, 8.0], [3.0, 7.0, 4.0, 1.0], [6.0, 2.0, 9.0, 3.0], [4.0, 8.0, 1.0, 5.0]]
NAN = float("nan")


def test_focal_statistics():
    w = [1, 5, 2, 3, 7, 4, 6, 2, 9]
    m = sum(w) / 9
    for fun, want in [
        ("mean", m),
        ("sum", 39.0),
        ("min", 1.0),
        ("max", 9.0),
        ("range", 8.0),
        ("median", 4.0),
        ("var", sum((x - m) ** 2 for x in w) / 8),
        ("sd", math.sqrt(sum((x - m) ** 2 for x in w) / 8)),
    ]:
        assert focal_statistics(G, fun=fun)[1][1] == pytest.approx(want, abs=1e-13)
    assert math.isnan(focal_statistics(G)[0][0])
    corner = focal_statistics(G, fun="mean", na_rm=True)[0][0]
    assert corner == pytest.approx((1 + 5 + 3 + 7) / 4)
    e = focal_statistics([[1, 1, 2], [1, 2, 2], [3, 3, 3]], fun="entropy")[1][1]
    p = [3 / 9, 3 / 9, 3 / 9]
    assert e == pytest.approx(-sum(q * math.log(q) for q in p))
    with pytest.raises(ValueError):
        focal_statistics(G, size=2)


def test_kernels_and_filters():
    for name in ("gaussian", "binomial", "mean"):
        assert sum(map(sum, filter_kernel(name, size=5, sigma=1.3))) == pytest.approx(1.0)
    g = filter_kernel("gaussian", size=3, sigma=1.0)
    assert g[0][0] / g[1][1] == pytest.approx(math.exp(-1.0))
    gb = filter_kernel("gabor", size=5, sigma=2.0, theta=0.0, wavelength=4.0, gamma=0.5)
    assert gb[2][3] == pytest.approx(math.exp(-1 / 8) * math.cos(math.pi / 2), abs=1e-15)
    assert gb[1][2] == pytest.approx(math.exp(-0.25 / 8), abs=1e-15)
    lap = focal_filter(G, filter_kernel("laplacian"))[1][1]
    assert lap == pytest.approx(5 + 3 + 4 + 2 - 4 * 7)
    sh = focal_filter(G, filter_kernel("sharpen"))[2][2]
    assert sh == pytest.approx(5 * 9 - 4 - 2 - 3 - 1)
    py = focal_filter(G, filter_kernel("prewitt_y"))[1][1]
    assert py == pytest.approx((6 + 2 + 9) - (1 + 5 + 2))
    nanG = [[1, 2, 3], [4, None, 6], [7, 8, 9]]
    assert math.isnan(focal_filter(nanG, filter_kernel("mean"))[1][1])
    assert focal_filter(nanG, filter_kernel("mean"), na_rm=True)[1][1] == pytest.approx(40 / 9)
    with pytest.raises(ValueError):
        filter_kernel("bogus")


def test_canny_and_morphology():
    step = [[0.0] * 4 + [10.0] * 4 for _ in range(8)]
    E = canny_edges(step, sigma=0.8, size=3)
    # symmetric step: the gradient maxima tie at columns 3 and 4, both kept; nothing else fires
    assert all(r == [0, 0, 0, 1, 1, 0, 0, 0] for r in E[1:7])
    assert canny_edges([[5.0] * 6] * 6) == [[0] * 6] * 6
    B = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
    assert grey_morphology(B, "closing")[2][2] == 1.0
    assert grey_morphology(B, "erosion")[2][2] == 0.0
    op = grey_morphology(B, "opening")
    assert all(v == 0.0 for r in op for v in r)
    with pytest.raises(ValueError):
        grey_morphology(B, "bogus")


def test_aggregate_resample_zonal_mask_distance_cost():
    assert raster_aggregate(G, 2, "max") == [[7.0, 8.0], [8.0, 9.0]]
    assert math.isnan(raster_aggregate([[1, None], [3, 4]], 2)[0][0])
    assert raster_aggregate([[1, None], [3, 4]], 2, na_rm=True) == [[8 / 3]]
    assert raster_disaggregate([[1], [2]], 3)[3] == [2.0, 2.0, 2.0]
    r = raster_resample(G, (0, 4, 0, 4), [1.0, 1.25, 0.2], [3.0, 2.75])
    assert r[0][0] == pytest.approx((1 + 5 + 3 + 7) / 4)
    # (1.25, 2.75): fx = 0.75, fy = 0.75 -> tx = 0.75, ty = 0.75 between centres of rows 0-1, columns 0-1
    assert r[1][1] == pytest.approx(0.25 * (0.25 * 1 + 0.75 * 5) + 0.75 * (0.25 * 3 + 0.75 * 7), abs=1e-14)
    assert math.isnan(r[0][2])
    assert raster_resample(G, (0, 4, 0, 4), [3.9], [0.1], method="near") == [[5.0]]
    z = zonal_statistics(G, [[1, 1, 2, 2]] * 4, "sum")
    assert (z.zone, z.value, z.count) == ([1.0, 2.0], [36.0, 33.0], [8, 8])
    m = raster_mask(G, [[1, 0, 1, 0]] * 4, maskvalue=0)
    assert math.isnan(m[0][1]) and m[0][0] == 1.0
    inv = raster_mask(G, [[1, 0, 1, 0]] * 4, maskvalue=0, inverse=True)
    assert math.isnan(inv[0][0]) and inv[0][1] == 5.0
    d = distance_transform([[None, None, None], [None, 1, None], [None, None, 5]], res=2.0)
    assert d[0][0] == pytest.approx(2 * math.sqrt(2)) and d[0][2] == pytest.approx(2 * math.sqrt(2))
    assert d[2][0] == pytest.approx(2 * math.sqrt(2)) and d[1][1] == 0.0 and d[2][1] == 2.0
    c = cost_distance([[1, 1, 1], [1, None, 1], [1, 1, 1]], [(0, 0)], res=10.0)
    assert math.isnan(c[1][1])
    # around the barrier: two straight steps and one diagonal, friction 1 everywhere
    assert c[2][2] == pytest.approx(10 * (2 + math.sqrt(2)))
    assert c[0][2] == pytest.approx(20.0) and c[2][0] == pytest.approx(20.0)
