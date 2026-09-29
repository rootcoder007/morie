"""Tests for specdist: MaxEnt species distributions and k-LoCoH home ranges."""

import math

from morie.fn.specdist import locoh_home_range, maxent_features, maxent_fit, maxent_predict

BG = [[math.sin(i * 0.37) * 2, math.cos(i * 0.91) + 0.01 * i] for i in range(120)]
PR = [[0.8 + 0.3 * math.sin(i), 0.5 + 0.2 * math.cos(i * 1.3)] for i in range(25)]


def test_features_scaling():
    f = maxent_features([[0.0, 1.0], [2.0, 3.0], [1.0, 2.0]], classes="lqph", hinge_knots=1)
    assert f.features[2][:4] == [0.5, 0.5, 0.25, 0.25]
    assert f.features[1][4] == 1.0  # product of the two scaled maxima
    assert f.features[2][5:] == [0.0, 0.0, 0.0, 0.0]


def test_maxent_moment_matching_and_outputs():
    fb = maxent_features(BG, classes="lq").features
    fp = maxent_features(PR, classes="lq", ranges=maxent_features(BG).ranges).features
    r = maxent_fit(fp, fb)
    B = fb + fp
    eta = [sum(a * b for a, b in zip(row, r.lambdas)) for row in B]
    z = sum(math.exp(v) for v in eta)
    q = [math.exp(v) / z for v in eta]
    for j in range(4):
        assert abs(sum(q[i] * B[i][j] for i in range(len(B))) - sum(row[j] for row in fp) / len(fp)) <= 1e-9
    raw = maxent_predict(r, B[:3], output="raw")
    assert max(abs(a - b) for a, b in zip(raw, q[:3])) <= 1e-12
    H = math.exp(-sum(v * math.log(v) for v in q))
    assert abs(maxent_predict(r, B[:1])[0] - (1 - math.exp(-H * q[0]))) <= 1e-12


def test_maxent_l1_kkt_conditions():
    fb = maxent_features(BG, classes="lq").features
    fp = maxent_features(PR, classes="lq", ranges=maxent_features(BG).ranges).features
    r = maxent_fit(fp, fb, l1=0.02)
    B = fb + fp
    eta = [sum(a * b for a, b in zip(row, r.lambdas)) for row in B]
    z = sum(math.exp(v) for v in eta)
    q = [math.exp(v) / z for v in eta]
    for j in range(4):
        g = sum(q[i] * B[i][j] for i in range(len(B))) - sum(row[j] for row in fp) / len(fp)
        if r.lambdas[j] != 0:
            assert abs(g + 0.02 * math.copysign(1, r.lambdas[j])) <= 1e-7
        else:
            assert abs(g) <= 0.02 + 1e-9


def test_locoh_square_union():
    pts = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 1), (10, 10), (10.5, 10), (10, 10.5)]
    r = locoh_home_range(pts, 3, levels=(1.0,))
    # the far triangle has area 0.125; the four square triangles union to the 4 x 1 diamond pieces
    assert r.coverage[0] == 1.0
    assert abs(min(r.hull_areas) - 0.125) <= 1e-12
    two = locoh_home_range([(0, 0), (1, 0), (0, 1), (1, 1)], 4, levels=(1.0,))
    assert two.areas == [1.0]
