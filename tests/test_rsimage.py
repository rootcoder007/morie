"""Tests for morie.fn.rsimage (multispectral image processing)."""

import math

import pytest

from morie.fn import rsimage as R

B1 = [[1.0, 2.0, 1.5, 3.0], [2.0, 2.5, 3.5, 4.0], [3.0, 3.2, 4.1, 5.0]]
B2 = [[0.5, 1.1, 0.9, 1.4], [1.2, 1.1, 1.9, 2.2], [1.4, 1.8, 2.0, 2.9]]
B3 = [[0.9, 0.7, 1.2, 1.0], [0.4, 0.8, 1.1, 1.3], [1.6, 0.9, 1.4, 1.7]]


def test_tasseled_cap_and_pca_reconstruction():
    r = R.tasseled_cap([[[0.1]], [[0.1]], [[0.1]], [[0.3]], [[0.2]], [[0.1]]], "landsat8oli")
    assert r.brightness[0][0] == pytest.approx(0.1 * (0.3029 + 0.2786 + 0.4733 + 0.1872) + 0.3 * 0.5599 + 0.2 * 0.508)
    p = R.band_pca([B1, B2, B3])
    for i in range(3):
        for j in range(4):
            x = [sum(p.scores[c][i][j] * p.loadings[c][b] for c in range(3)) + p.center[b] for b in range(3)]
            assert x == pytest.approx([B1[i][j], B2[i][j], B3[i][j]], abs=1e-12)
    assert R.band_pca([[[1, 2], [3, 4]], [[2, 4], [6, 8]]]).sdev[0] == pytest.approx(math.sqrt(5 * 5 / 3), abs=1e-12)


def test_mnf_unit_noise():
    m = R.mnf_transform([B1, B2, B3])
    L, N = m.loadings, m.noise_cov
    for a in range(3):
        for b in range(3):
            v = sum(L[a][i] * N[i][j] * L[b][j] for i in range(3) for j in range(3))
            assert v == pytest.approx(1.0 if a == b else 0.0, abs=1e-9)
    assert m.snr == sorted(m.snr, reverse=True)


def test_pan_sharpen():
    assert [b[0][0] for b in R.pan_sharpen([[[1.0]], [[2.0]], [[1.0]]], [[8.0]]).bands] == [2.0, 4.0, 2.0]
    intensity = [[(a + b + c) / 3 for a, b, c in zip(r1, r2, r3)] for r1, r2, r3 in zip(B1, B2, B3)]
    ihs = R.pan_sharpen([B1, B2, B3], intensity, method="ihs").bands
    assert ihs[0] == [pytest.approx(row, abs=1e-12) for row in B1]
    assert ihs[2] == [pytest.approx(row, abs=1e-12) for row in B3]


def test_topographic_correction():
    flat = R.topographic_correction(
        [B1], [[0.0] * 4] * 3, [[0.0] * 4] * 3, sun_azimuth=2.0, sun_zenith=0.5, method="cos"
    )
    assert flat.bands[0] == [pytest.approx(row, abs=1e-12) for row in B1]
    S = [[0.1 * (i + j) for j in range(4)] for i in range(3)]
    A = [[0.5 * j for j in range(4)] for _ in range(3)]
    c = R.topographic_correction([B1], S, A, sun_azimuth=2.0, sun_zenith=0.5, method="C")
    il = [v for row in c.illumination for v in row]
    x = [v for row in B1 for v in row]
    mi, mx = sum(il) / 12, sum(x) / 12
    m = sum((a - mi) * (b - mx) for a, b in zip(il, x)) / sum((a - mi) ** 2 for a in il)
    ck = (mx - m * mi) / m
    assert c.bands[0][1][2] == pytest.approx(B1[1][2] * (math.cos(0.5) + ck) / (c.illumination[1][2] + ck))


def test_radiometry_and_masks():
    assert R.radiometric_correction([[[100.0]]], [0.5], [1.0], method="rad").bands[0][0][0] == 51.0
    ref = R.radiometric_correction(
        [[[200.0]]], [0.01], [-0.1], method="apref", esun=[1800.0], sun_elevation=40, distance=1.0
    ).bands[0][0][0]
    assert ref == pytest.approx(math.pi * (0.01 * 200 - 0.1) / (1800 * math.cos(50 * math.pi / 180)))
    assert R.estimate_haze([[0, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10, 12]], dark_prop=0.2, max_slope=False) == 5.0
    assert R.cloud_mask([[0.1, 0.9]], [[300.0, 250.0]]).mask == [[False, True]]
    assert R.cloud_shadow_mask([[True, False, False]], (1, 0)).mask == [[False, True, False]]


def test_classifiers_and_indices():
    assert R.spectral_angle_classify([[[1.0, 0.1]], [[0.1, 1.0]]], [[1, 0], [0, 1]]).classes == [[0, 1]]
    tr = [[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [5.0, 5.0], [5.2, 4.9], [4.8, 5.3]]
    ml = R.gaussian_ml_classify([[[0.1, 4.9]], [[0.2, 5.1]]], tr, ["w", "w", "w", "v", "v", "v"])
    assert ml.classes == [["w", "v"]] and sum(ml.posterior[0][0]) == pytest.approx(1.0)
    k = R.kmeans_classify([[[0.0, 0.2, 5.0, 5.2]]], [[1.0], [4.0]])
    assert k.classes == [[0, 0, 1, 1]] and k.centers == [[pytest.approx(0.1)], [pytest.approx(5.1)]]
    assert R.atgp_endmembers([[[1.0, 0.0, 0.5]], [[0.0, 2.0, 0.5]]], 2).index == [1, 0]
    f = R.fpar_from_ndvi([0.5], method="ndvi").fpar[0]
    assert f == pytest.approx((0.5 - 0.05) * (0.95 - 0.001) / 0.9 + 0.001)
    t = R.tvdi([0.1, 0.1, 0.9, 0.9], [300.0, 320.0, 295.0, 305.0], n_bins=2)
    assert t.tvdi == pytest.approx([0.2, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        R.tasseled_cap([[[0.1]]], "hubble")
