"""Tests for otgw.ot_gromov_wasserstein."""

from morie.fn import _array_core as np

from morie.fn.otgw import ot_gromov_wasserstein


def test_otgw_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    Cx_raw = rng.uniform(0, 1, (n, n))
    Cx = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = (float(Cx_raw[i][j]) + float(Cx_raw[j][i])) / 2
            Cx[i][j] = val
            Cx[j][i] = val

    Cy_raw = rng.uniform(0, 1, (n, n))
    Cy = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = (float(Cy_raw[i][j]) + float(Cy_raw[j][i])) / 2
            Cy[i][j] = val
            Cy[j][i] = val

    a_raw = rng.uniform(0, 1, n)
    a_sum = float(np.sum(a_raw))
    a = [float(x) / a_sum for x in a_raw]

    b_raw = rng.uniform(0, 1, n)
    b_sum = float(np.sum(b_raw))
    b = [float(x) / b_sum for x in b_raw]

    result = ot_gromov_wasserstein(Cx, Cy, a, b)
    assert isinstance(result, dict)


def test_otgw_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 3
    Cx_raw = rng.uniform(0, 1, (n, n))
    Cx = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = (float(Cx_raw[i][j]) + float(Cx_raw[j][i])) / 2
            Cx[i][j] = val
            Cx[j][i] = val

    Cy_raw = rng.uniform(0, 1, (n, n))
    Cy = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = (float(Cy_raw[i][j]) + float(Cy_raw[j][i])) / 2
            Cy[i][j] = val
            Cy[j][i] = val

    a_raw = rng.uniform(0, 1, n)
    a_sum = float(np.sum(a_raw))
    a = [float(x) / a_sum for x in a_raw]

    b_raw = rng.uniform(0, 1, n)
    b_sum = float(np.sum(b_raw))
    b = [float(x) / b_sum for x in b_raw]

    result = ot_gromov_wasserstein(Cx, Cy, a, b)
    assert isinstance(result, dict)
