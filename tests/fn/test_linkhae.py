"""Tests for linkhae (Haebara characteristic-curve linking).

Replaces the generated stub, which imported ``linking_haebara``.
"""

from morie.fn.linkhae import linkhae


def _items(a=None, b=None, c=None):
    a = a or [1.0, 0.8, 1.2, 0.9, 1.1]
    b = b or [-1.0, -0.5, 0.0, 0.5, 1.0]
    c = c or [0.0] * len(a)
    return [(a[i], b[i], c[i]) for i in range(len(a))]


def _rescale(items, A, B):
    return [(it[0] / A, A * it[1] + B, it[2]) for it in items]


def test_a_known_rescaling_is_recovered():
    src = _items()
    A, B = 1.6, -0.4
    res = linkhae(src, _rescale(src, A, B))
    assert abs(res["A"] - A) < 1e-3
    assert abs(res["B"] - B) < 1e-3


def test_the_identity_link_gives_A_1_B_0():
    src = _items()
    res = linkhae(src, src)
    assert abs(res["A"] - 1.0) < 1e-4
    assert abs(res["B"]) < 1e-4
    assert res["criterion"] < 1e-8


def test_the_criterion_is_minimised_at_the_solution():
    src = _items()
    A, B = 1.4, 0.3
    res = linkhae(src, _rescale(src, A, B))
    assert res["criterion"] < 1e-6
    assert res["n_common"] == 5


def test_the_symmetric_variant_runs_and_agrees_closely():
    src = _items()
    tgt = _rescale(src, 1.5, 0.2)
    one = linkhae(src, tgt)
    sym = linkhae(src, tgt, symmetric=True)
    assert sym["symmetric"] is True
    assert abs(sym["A"] - one["A"]) < 0.1


def test_validation():
    src = _items()
    for call in (lambda: linkhae(src[:1], src[:1]),
                 lambda: linkhae(src, src[:-1])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
