"""forestry: inventory formulas and raster algorithms checked against their definitions."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.forestry import (
    adaptive_cluster_estimate,
    basal_area,
    canopy_gaps,
    canopy_height_model,
    carbon_stock,
    crown_segmentation,
    line_intersect_volume,
    plot_estimate,
    rasterize_points,
    stand_density_index,
    stratified_estimate,
    tree_biomass,
    tree_tops,
)

U = [float(v) for v in random_uniform(800, seed=3)]


def test_stand_measures():
    d = [12.0, 25.0, 31.5, 18.2]
    e = [50.0, 30.0, 20.0, 40.0]
    ba = basal_area(d, e)
    assert ba.total == pytest.approx(sum(math.pi * x * x / 40000 * w for x, w in zip(d, e)), rel=1e-14)
    s = stand_density_index(d, e)
    qmd = math.sqrt(sum(w * x * x for x, w in zip(d, e)) / 140.0)
    assert s.qmd == pytest.approx(qmd, rel=1e-14)
    assert s.sdi == pytest.approx(140.0 * (qmd / 25.4) ** 1.605, rel=1e-13)
    same = [22.0] * 5
    assert stand_density_index(same, 80.0).sdi == pytest.approx(
        stand_density_index(same, 80.0, "summation").sdi, rel=1e-13
    )
    b = tree_biomass([35.0], [27.0], 0.55)[0]
    assert b == pytest.approx(0.0673 * (0.55 * 35.0**2 * 27.0) ** 0.976, rel=1e-14)
    b7 = tree_biomass([35.0], wood_density=0.55, stress=-0.1)[0]
    ld = math.log(35.0)
    assert b7 == pytest.approx(
        math.exp(-1.803 + 0.0976 + 0.976 * math.log(0.55) + 2.673 * ld - 0.0299 * ld * ld), rel=1e-13
    )
    c = carbon_stock([b, b7], 0.5, 0.2)
    assert c.co2e == pytest.approx((b + b7) * 0.5 * 1.2 * 44 / 12, rel=1e-13)
    assert line_intersect_volume([8.0, 12.0], 50.0) == pytest.approx(math.pi**2 * 208 / 400, rel=1e-14)


def test_rasterize_matches_brute_force():
    x = [10 * v for v in U[:150]]
    y = [7 * v for v in U[150:300]]
    z = U[300:450]
    r = rasterize_points(x, y, z, 1.5, "max")
    xmin, ymax, nc, nr = r.extent
    for i in range(nr):
        for j in range(nc):
            inside = [
                zz
                for xx, yy, zz in zip(x, y, z)
                if xmin + j * 1.5 <= xx < xmin + (j + 1) * 1.5 and ymax - (i + 1) * 1.5 < yy <= ymax - i * 1.5
            ]
            if inside:
                assert r.grid[i][j] == max(inside)
            else:
                assert math.isnan(r.grid[i][j])
    assert canopy_height_model([[5.0, math.nan]], [[6.0, 1.0]]) == [
        [0.0, canopy_height_model([[5.0, math.nan]], [[6.0, 1.0]])[0][1]]
    ]


def test_gaps_and_connectivity():
    chm = [[9, 1, 9, 9], [9, 9, 1, 9], [1, 1, 9, 9], [9, 9, 9, 1]]
    g8 = canopy_gaps(chm, 5.0)
    g4 = canopy_gaps(chm, 5.0, connectivity=4)
    # the two gaps of the upper left touch diagonally: one gap of 4 cells with 8-connectivity
    assert sorted(g8.areas) == [1.0, 4.0]
    assert sorted(g4.areas) == [1.0, 1.0, 1.0, 2.0]
    assert g8.gap_fraction == pytest.approx(5 / 16, rel=1e-15)
    big = canopy_gaps(chm, 5.0, min_area=8.0, res=2.0)
    assert big.areas == [16.0]


def test_tree_tops_are_window_maxima():
    chm = [[20 * U[i * 12 + j] for j in range(12)] for i in range(10)]
    t = tree_tops(chm, 1.0, 5.0, window=4.0)
    for i in range(10):
        for j in range(12):
            h = chm[i][j]
            is_max = h >= 5.0 and all(
                chm[a][b] <= h for a in range(10) for b in range(12) if (a - i) ** 2 + (b - j) ** 2 <= 4.0
            )
            assert ([i, j] in t.cells) == is_max


def test_crowns_satisfy_growth_rules():
    chm = [
        [
            17 * math.exp(-((i - 5) ** 2 + (j - 6) ** 2) / 8)
            + 12 * math.exp(-((i - 11) ** 2 + (j - 13) ** 2) / 10)
            + U[i * 18 + j]
            for j in range(18)
        ]
        for i in range(16)
    ]
    s = crown_segmentation(chm, smooth=False, th=2.0)
    assert len(s.tops) >= 2
    for k, (a, b) in enumerate(s.tops, start=1):
        assert s.labels[a][b] == k
        hs = chm[a][b]
        cells = [(i, j) for i in range(16) for j in range(18) if s.labels[i][j] == k]
        assert len(cells) == s.crown_cells[k - 1]
        assert s.mean_height[k - 1] == pytest.approx(sum(chm[i][j] for i, j in cells) / len(cells), rel=1e-12)
        for i, j in cells:
            assert chm[i][j] <= 1.05 * hs + 1e-12 and chm[i][j] > 0.45 * hs
            assert math.hypot(i - a, j - b) < 10


def test_estimators():
    y = [4.0, 7.0, 5.0, 9.0, 6.0]
    p = plot_estimate(y, 0.05, 200)
    per = [v / 0.05 for v in y]
    m = sum(per) / 5
    assert p.mean == pytest.approx(m, rel=1e-14)
    assert p.se == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in per) / 4 / 5 * (1 - 5 / 200)), rel=1e-13)
    st = stratified_estimate([2.0, 4.0, 3.0, 10.0, 12.0], ["u", "u", "u", "d", "d"], {"u": 30, "d": 10})
    assert st.mean == pytest.approx(0.75 * 3 + 0.25 * 11, rel=1e-14)
    assert st.se == pytest.approx(math.sqrt(0.75**2 * 1 / 3 * 0.9 + 0.25**2 * 2 / 2 * 0.8), rel=1e-13)
    singles = [[v] for v in y]
    ac = adaptive_cluster_estimate(singles, 40)
    assert ac.mean_hh == pytest.approx(sum(y) / 5, rel=1e-14)
    assert ac.mean_ht == pytest.approx(sum(y) / 5, rel=1e-12)
    net = adaptive_cluster_estimate([[0.0], [5.0, 3.0], [5.0, 3.0]], 30, ids=["a", "b", "b"])
    alpha = 1 - math.comb(28, 3) / math.comb(30, 3)
    assert net.mean_ht == pytest.approx(8.0 / alpha / 30, rel=1e-12)
