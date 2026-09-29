"""Tests for morie.fn.fdadj: the front-door sum recomputed by counting."""

from morie.fn.fdadj import frontdoor_adjustment

x = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]
z = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
y = [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1]


def _fd(t):
    """sum_z P(z|t) sum_x P(y=1|x,z) P(x), by counting."""
    n = len(x)
    out = 0.0
    for zv in (0, 1):
        pz = sum(1 for i in range(n) if x[i] == t and z[i] == zv) / sum(1 for i in range(n) if x[i] == t)
        inner = 0.0
        for xv in (0, 1):
            cell = [y[i] for i in range(n) if x[i] == xv and z[i] == zv]
            inner += sum(cell) / len(cell) * sum(1 for v in x if v == xv) / n
        out += pz * inner
    return out


def test_front_door_sum():
    d = frontdoor_adjustment(x, z, y)["distribution"]
    for t in (0, 1):
        assert abs(d[t][1] - _fd(t)) < 1e-15
        assert abs(d[t][0] + d[t][1] - 1) < 1e-15
