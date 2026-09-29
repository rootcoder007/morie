"""Tests for morie.fn.elbdm -- elbow method MDS dimensionality."""

from morie.fn.elbdm import elbdm, elbow_mds_dim


def test_elbdm_smoke():
    stresses = [0.3, 0.15, 0.08, 0.06, 0.055, 0.054]
    r = elbdm(stresses)
    assert r.name == "elbow_mds_dim"
    assert 1 <= r.value <= 6


def test_elbdm_short():
    r = elbdm([0.5, 0.1])
    assert r.value == 1


def test_elbdm_alias():
    assert elbdm is elbow_mds_dim


def test_knee_is_the_farthest_point_from_the_chord():
    import math

    s = [0.30, 0.15, 0.08, 0.06, 0.055, 0.054]
    n = len(s)
    x1, y1, x2, y2 = 1.0, s[0], float(n), s[-1]
    L = math.hypot(x2 - x1, y2 - y1)
    d = [abs((x2 - x1) * (y1 - s[i]) - (y2 - y1) * (x1 - (i + 1))) / L for i in range(n)]
    assert elbow_mds_dim(s).value == d.index(max(d)) + 1
