import itertools
import math

from morie.fn.infocodes import (
    bounded_distance_noise,
    code_length_decomposition,
    mcgill_interaction_information,
    repetition_error_approx,
    robust_soliton,
    runlength_channel_capacity,
    self_dual_code_check,
)


def test_bounded_distance_noise():
    for R in (0.25, 0.5, 0.9):
        r = bounded_distance_noise(R)
        f = r.f_shannon
        assert abs(-f * math.log2(f) - (1 - f) * math.log2(1 - f) - (1 - R)) < 1e-12
        assert r.f_bd == f / 2


def test_self_dual():
    for perm in itertools.permutations(range(4)):
        P = [[1 if perm[i] == j else 0 for j in range(4)] for i in range(4)]
        r = self_dual_code_check(P)
        assert r.self_dual and all(v == 0 for row in r.GGt for v in row)
    r = self_dual_code_check([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert not r.self_dual and any(v for row in r.GGt for v in row)


def test_runlength_capacity():
    assert abs(runlength_channel_capacity(1).capacity - math.log2((1 + math.sqrt(5)) / 2)) < 1e-14
    for L in (2, 3, 6):
        r = runlength_channel_capacity(L)
        assert abs(sum(2.0 ** (-r.capacity * ell) for ell in range(1, L + 2)) - 1) < 1e-13
        assert r.simple_rate < r.capacity < 1


def test_repetition_and_soliton():
    f, N = 0.1, 7
    r = repetition_error_approx(N, f, target=1e-15)
    assert abs(r.pb - (4 * f * (1 - f)) ** (N / 2)) < 1e-15 and round(r.n_target) == 68
    K, c, d = 10000, 0.2, 0.05
    s = robust_soliton(K, c, d)
    S = c * math.log(K / d) * math.sqrt(K)
    assert abs(s.S - S) < 1e-12 and s.m == 41 and round(S) == 244
    Z = 1 + S / K * sum(1 / j for j in range(1, s.m)) + S * math.log(S / d) / K
    assert abs(s.Z - Z) < 1e-12
    assert abs(sum(s.mu) - 1) < 1e-12


def test_code_length_identity():
    p, ln = [0.3, 0.3, 0.2, 0.1, 0.1], [2, 2, 2, 3, 4]
    r = code_length_decomposition(p, ln)
    assert abs(r.L - sum(a * b for a, b in zip(p, ln))) < 1e-15
    assert abs(r.L - (r.H + r.kl - math.log2(r.kraft))) < 1e-13
    assert abs(r.H + sum(v * math.log2(v) for v in p)) < 1e-14


def _H(ps):
    return -sum(v * math.log2(v) for v in ps if v > 0)


def test_interaction_information_entropies():
    T = [[[3, 1, 2], [0, 4, 1]], [[2, 2, 5], [1, 3, 2]], [[4, 0, 1], [2, 1, 6]]]
    n = sum(v for a in T for b in a for v in b)
    P = [[[v / n for v in b] for b in a] for a in T]
    NI, J, K = 3, 2, 3
    hx = _H([sum(P[i][j][k] for j in range(J) for k in range(K)) for i in range(NI)])
    hy = _H([sum(P[i][j][k] for i in range(NI) for k in range(K)) for j in range(J)])
    hz = _H([sum(P[i][j][k] for i in range(NI) for j in range(J)) for k in range(K)])
    hxy = _H([sum(P[i][j]) for i in range(NI) for j in range(J)])
    hxz = _H([sum(P[i][j][k] for j in range(J)) for i in range(NI) for k in range(K)])
    hyz = _H([sum(P[i][j][k] for i in range(NI)) for j in range(J) for k in range(K)])
    hxyz = _H([v for a in P for b in a for v in b])
    ii = hxy + hxz + hyz - hxyz - hx - hy - hz
    assert abs(mcgill_interaction_information(T).ii - ii) < 1e-12
    xor = [[[0.25 if (i ^ j) == k else 0.0 for k in (0, 1)] for j in (0, 1)] for i in (0, 1)]
    assert abs(mcgill_interaction_information(xor).ii - 1) < 1e-14
