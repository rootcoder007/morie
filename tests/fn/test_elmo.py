"""Tests for elmo. Full anchor: ledger/wave3/anchor_nlp_family.py."""
import pytest
from morie.fn import _array_core as np
from morie.fn.elmo import elmo_mix, elmo_representation, layer_weights

REPS = [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]


def test_the_weights_are_a_simplex_and_gamma_carries_the_scale():
    """s chooses WHICH layers to read and cannot scale the output; all
    magnitude is gamma's. Free s would make gamma unidentifiable."""
    s = layer_weights([1.0, 2.0, 3.0])
    assert sum(s) == pytest.approx(1.0, abs=1e-15)
    assert all(v > 0 for v in s)
    m1 = elmo_mix(REPS, [0.0, 0.0, 0.0], gamma=1.0)
    m2 = elmo_mix(REPS, [0.0, 0.0, 0.0], gamma=2.0)
    for c in range(2):
        assert m2[0][c] == pytest.approx(2.0 * m1[0][c], abs=1e-15)


def test_special_cases_of_equation_one():
    top = elmo_mix(REPS, [-50.0, -50.0, 50.0])
    assert top[0] == pytest.approx(REPS[2][0], abs=1e-9)
    flat = elmo_mix(REPS, [0.0, 0.0, 0.0])
    for c in range(2):
        assert flat[0][c] == pytest.approx(
            sum(REPS[j][0][c] for j in range(3)) / 3.0, abs=1e-15)
    with pytest.raises(ValueError):
        elmo_mix(REPS, [0.0, 0.0])


def test_layer_zero_is_the_token_vector_duplicated():
    """h_{k,0} = [x_k; x_k], so every layer is the same width and
    eq. (1) is well defined."""
    d = 3
    rng = np.random.default_rng(1)
    X = [[rng.standard_normal() for _ in range(d)] for _ in range(4)]

    def layer(sd):
        r = np.random.default_rng(sd)
        return ([[r.standard_normal() * 0.3 for _ in range(4 * d)]
                 for _ in range(d)],
                [[r.standard_normal() * 0.3 for _ in range(4 * d)]
                 for _ in range(d)], [0.0] * (4 * d))

    lay = layer(1) + layer(2)
    r = elmo_representation(X, [lay], raw_weights=[0.0, 0.0])
    assert r["n_layers"] == 2
    for t in range(4):
        for c in range(d):
            assert r["layers"][0][t][c] == pytest.approx(X[t][c])
            assert r["layers"][0][t][d + c] == pytest.approx(X[t][c])
    # forward and backward halves must differ, or one direction is dead
    assert any(abs(r["layers"][1][t][c] - r["layers"][1][t][d + c])
               > 1e-9 for t in range(4) for c in range(d))
    with pytest.raises(ValueError):
        elmo_representation([[1.0, 2.0]] * 4, [lay],
                            raw_weights=[0.0, 0.0])
