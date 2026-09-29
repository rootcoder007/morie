"""Tests for morie.fn.medFront: argument order (Y, X, M) of the front-door formula."""

from morie.fn.medFront import front_door

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
    d = front_door(y, x, z)["distribution"]
    assert abs(d[1][1] - _fd(1)) < 1e-15
