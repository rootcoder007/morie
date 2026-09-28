import math

import pytest

from morie.fn.symalg import galois_group, jordan_canonical, shunting_yard


def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _det(M):
    M = [list(map(float, r)) for r in M]
    n, d = len(M), 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if M[p][c] == 0:
            return 0.0
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        for i in range(c + 1, n):
            f = M[i][c] / M[c][c]
            M[i] = [M[i][j] - f * M[c][j] for j in range(n)]
    return d


@pytest.mark.parametrize(
    "A",
    [
        [[5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]],
        [[2, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 2, 1, 0], [0, 0, 0, 2, 1], [0, 0, 0, 0, 2]],
        [[1, 2, 0], [-2, -3, 0], [4, 4, -1]],
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
    ],
)
def test_jordan_similarity(A):
    r = jordan_canonical(A)
    assert _mm(A, r.P) == _mm(r.P, r.J)
    assert abs(_det(r.P)) > 0.5
    assert sum(s for _, s in r.blocks) == len(A)


def test_jordan_block_sizes_and_errors():
    r = jordan_canonical([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    assert r.blocks == [(0, 4)]
    with pytest.raises(ValueError):
        jordan_canonical([[0, -1], [1, 0]])
    with pytest.raises(ValueError):
        jordan_canonical([[0.5, 0], [0, 1]])


@pytest.mark.parametrize(
    "poly,group",
    [
        ([1, 0, 0, 0, -2], "D4"),
        ([1, 0, 0, 0, 1], "V4"),
        ([1, 1, 1, 1, 1], "C4"),
        ([1, 0, 0, -1, -1], "S4"),
        ([1, 0, 0, 8, 12], "A4"),
        ([1, 0, -3, 0, 1], "reducible"),
        ([1, 0, 0, -2], "S3"),
        ([1, -3, 0, 1], "A3"),
        ([1, 0, -2], "C2"),
    ],
)
def test_galois_known_groups(poly, group):
    assert galois_group(poly).group == group


def test_galois_discriminant_formula():
    g = galois_group([1, -3, 0, 1])
    a, b, c = -3, 0, 1
    assert g.discriminant == a * a * b * b - 4 * b**3 - 4 * a**3 * c - 27 * c * c + 18 * a * b * c
    assert math.isqrt(g.discriminant) ** 2 == g.discriminant


def test_shunting_yard():
    r = shunting_yard("3 + 4 * 2 / (1 - 5) ^ 2 ^ 3")
    assert r.value == pytest.approx(3 + 4 * 2 / (1 - 5) ** (2**3), abs=1e-12)
    assert shunting_yard("-2^2").value == -4.0
    r = shunting_yard("max(1, x) * sin(y) - 2^-3", {"x": 3, "y": 0.5})
    assert r.value == pytest.approx(3 * math.sin(0.5) - 2**-3, abs=1e-12)
    assert shunting_yard("a + b").value is None
    with pytest.raises(ValueError):
        shunting_yard("(1 + 2")
