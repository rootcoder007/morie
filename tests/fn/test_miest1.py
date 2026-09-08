"""Tests for miest1 (Kraskov k-nearest-neighbour mutual information).

Replaces the generated stub, which imported a name the module never had.
"""

import math

from morie.fn.miest1 import miest1


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def test_independent_variables_have_mutual_information_near_zero():
    r = _lcg(3)
    X = [_gauss(r) for _ in range(400)]
    Y = [_gauss(r) for _ in range(400)]
    res = miest1(X, Y, k=3)
    assert abs(res["mi"]) < 0.1


def test_a_correlated_pair_scores_close_to_the_closed_form():
    # for a bivariate normal, I = -0.5 log(1 - rho^2)
    rho = 0.8
    r = _lcg(5)
    X, Y = [], []
    for _ in range(600):
        a, b = _gauss(r), _gauss(r)
        X.append(a)
        Y.append(rho * a + math.sqrt(1 - rho * rho) * b)
    want = -0.5 * math.log(1 - rho * rho)
    assert abs(miest1(X, Y, k=3)["mi"] - want) < 0.15


def test_stronger_dependence_gives_more_information():
    r = _lcg(7)
    base = [_gauss(r) for _ in range(400)]
    noise = [_gauss(r) for _ in range(400)]
    weak = miest1(base, [0.2 * base[i] + noise[i]
                         for i in range(400)], k=3)["mi"]
    strong = miest1(base, [3.0 * base[i] + noise[i]
                           for i in range(400)], k=3)["mi"]
    assert strong > weak


def test_bits_are_nats_over_log_two():
    r = _lcg(11)
    X = [_gauss(r) for _ in range(200)]
    Y = [x + 0.5 * _gauss(r) for x in X]
    res = miest1(X, Y, k=3)
    assert abs(res["mi_bits"] - res["mi"] / math.log(2.0)) < 1e-12


def test_both_algorithms_run_and_broadly_agree():
    r = _lcg(13)
    X = [_gauss(r) for _ in range(300)]
    Y = [x + 0.7 * _gauss(r) for x in X]
    a1 = miest1(X, Y, k=4, algorithm=1)["mi"]
    a2 = miest1(X, Y, k=4, algorithm=2)["mi"]
    assert abs(a1 - a2) < 0.3


def test_validation():
    for call in (lambda: miest1([1.0], [1.0]),
                 lambda: miest1([1.0, 2.0], [1.0]),
                 lambda: miest1([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], k=0),
                 lambda: miest1([1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                                algorithm=3)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
