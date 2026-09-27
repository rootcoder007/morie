"""Nucleolus, prenucleolus (CoopGame, TUGLab) and kernel points of TU games."""

from morie.fn.tukern import kernel_point
from morie.fn.tunucl import nucleolus

G1 = [0, 0, 26.413, 0, 37.955, 39.934, 49.945, 0, 38.706, 23.507, 58.321, 31.357, 57.588, 58.409, 82.377]
G2 = [0, 0, 4.324, 0, 0.155, 6.061, 13.931, 0, 19.323, 5.85, 8.784, 4.403, 18.23, 17.584, 12.825]


def close(a, b, tol):
    return all(abs(u - w) <= tol for u, w in zip(a, b))


def test_nucleolus_matches_coopgame():
    # CoopGame::nucleolus / prenucleolus on the same games (lexicographic order)
    ref = [20.611666666667, 21.102833333333, 20.699666666667, 19.962833333333]
    assert close(nucleolus(G1).value, ref, 1e-9) and close(nucleolus(G1, pre=True).value, ref, 1e-9)
    assert close(nucleolus(G2).value, [3.637, 0.0, 1.898, 7.29], 1e-9)
    p = nucleolus(G2, pre=True)
    assert close(p.value, [3.637, -2.359, 4.257, 7.29], 1e-9)
    # first level = least-core value: the largest excess at the prenucleolus
    ex = [G2[S - 1] - sum(p.value[i] for i in range(4) if S >> i & 1) for S in range(1, 15)]
    assert abs(p.extra["levels"][0] - max(ex)) < 1e-9


def test_symmetric_game():
    r = nucleolus([0, 0, 60, 0, 60, 60, 72])
    assert close(r.value, [24, 24, 24], 1e-9) and close(r.extra["levels"], [12.0], 1e-9)


def test_kernel_equals_nucleolus_where_theory_says_so():
    # Maschler, Peleg and Shapley (1972): convex games have kernel = nucleolus
    w = [1.0, 2.0, 3.0, 5.0, 8.0]
    v = [sum(w[i] for i in range(5) if S >> i & 1) ** 2 for S in range(1, 32)]
    nu = nucleolus(v).value
    for x0 in (None, [60.0, 60.0, 60.0, 60.0, 121.0], [1.0, 4.0, 9.0, 25.0, 322.0]):
        k = kernel_point(v, x0)
        assert k.extra["converged"] and k.extra["in_kernel"] and close(k.value, nu, 1e-8)
    # three players: kernel = nucleolus for zero-monotonic games
    g = [0, 0, 26.0, 0, 38.0, 40.0, 50.0]
    assert close(kernel_point(g).value, nucleolus(g).value, 1e-8)
    assert close(kernel_point(g, pre=True).value, nucleolus(g, pre=True).value, 1e-8)


def test_nucleolus_lies_in_the_kernel():
    for v in (G1, G2):
        k = kernel_point(v, nucleolus(v).value)
        assert k.extra["iterations"] == 0 and k.extra["in_kernel"]
        s = k.extra["surplus"]
        x = k.value
        single = [v[(1 << i) - 1] for i in range(4)]
        assert all(s[i][j] <= s[j][i] + 1e-8 or x[j] - single[j] <= 1e-8 for i in range(4) for j in range(4) if i != j)
