"""Tests for mambss -- Mamba's selective SSM step (S6).

Replaces a generated test that called a stub returning mean(y). Full
anchor: ledger/wave3/anchor_mambss.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.mambss import (discretize_zoh, gated_rnn_equivalent,
                             selective_scan, selective_ssm_step,
                             softplus)


def test_softplus():
    for z in (-5.0, -1.0, 0.0, 1.0, 5.0):
        assert softplus(z) == pytest.approx(math.log(1.0 + math.exp(z)),
                                            abs=1e-14)
    # exp(-800) is below the smallest double, so this underflows to
    # exactly zero -- the point is that it does so without raising
    assert softplus(800.0) == pytest.approx(800.0, abs=1e-9)
    assert softplus(-800.0) == 0.0
    assert softplus(-100.0) > 0.0


def test_zero_order_hold():
    Abar, Bbar = discretize_zoh(0.7, [-1.0], [1.0])
    assert Abar[0] == pytest.approx(math.exp(-0.7), abs=1e-15)
    assert Bbar[0] == pytest.approx(1.0 - math.exp(-0.7), abs=1e-15)
    # the A -> 0 limit must be smooth, not 0/0
    _, B0 = discretize_zoh(0.3, [-1e-12], [2.0])
    assert B0[0] == pytest.approx(0.6, abs=1e-9)
    # and the euler rule really is the Delta*B simplification
    assert discretize_zoh(0.7, [-1.0], [1.0],
                          rule="euler")[1][0] == pytest.approx(0.7)


def test_theorem_one_holds_to_machine_precision():
    """With N=1, A=-1, B=1 and softplus, S6 IS a gated RNN. Delta =
    softplus(s) gives Abar = exp(-Delta) = 1 - sigmoid(s) and
    Bbar = 1 - exp(-Delta) = sigmoid(s), so this is an identity."""
    rng = np.random.default_rng(7)
    xs = [rng.standard_normal() for _ in range(40)]
    w, b = 1.37, -0.42
    res = selective_scan([[v] for v in xs], [[-1.0]], [[0.0]], [[0.0]],
                         [[w]], b_B=[1.0], b_C=[1.0], b_delta=b)
    h_rnn, g_rnn = gated_rnn_equivalent(xs, w, b)
    for t in range(40):
        assert res["y"][t][0] == pytest.approx(h_rnn[t], abs=1e-14)
    # the gate must genuinely vary or the identity is trivial
    assert max(g_rnn) - min(g_rnn) > 0.5


def test_the_delta_b_simplification_breaks_theorem_one():
    """A wrong discretisation is otherwise invisible: the model still
    runs, just not as the paper specifies."""
    rng = np.random.default_rng(7)
    xs = [rng.standard_normal() for _ in range(40)]
    w, b = 1.37, -0.42
    res = selective_scan([[v] for v in xs], [[-1.0]], [[0.0]], [[0.0]],
                         [[w]], b_B=[1.0], b_C=[1.0], b_delta=b,
                         rule="euler")
    h_rnn, _ = gated_rnn_equivalent(xs, w, b)
    gap = max(abs(res["y"][t][0] - h_rnn[t]) for t in range(40))
    assert gap > 1e-3


@pytest.mark.parametrize("s_raw", [-20.0, 20.0])
def test_the_gate_writes_in_exactly_g_times_x(s_raw):
    delta = softplus(s_raw)
    gate = 1.0 - math.exp(-delta)
    h, _ = selective_ssm_step(5.0, [0.0], [-1.0], [1.0], [1.0], delta)
    assert h[0] == pytest.approx(gate * 5.0, abs=1e-15)


def test_the_model_is_time_varying_and_can_be_switched_back():
    X = [[float(v)] for v in (0.0, 3.0, 0.0, -3.0, 0.0)]
    varying = selective_scan(X, [[-1.0]], [[0.4]], [[0.6]], [[1.1]],
                             b_B=[0.2], b_C=[0.3])
    d = [varying["delta"][t][0] for t in range(len(X))]
    assert max(d) - min(d) > 0.5
    assert varying["time_invariant"] is False
    # with W_delta = 0 it collapses to a constant Delta, which is S4
    const = selective_scan(X, [[-1.0]], [[0.4]], [[0.6]], [[0.0]],
                           b_B=[0.2], b_C=[0.3])
    dc = [const["delta"][t][0] for t in range(len(X))]
    assert max(dc) - min(dc) < 1e-15


def test_the_step_reproduces_the_recurrence():
    rng = np.random.default_rng(7)
    xs = [rng.standard_normal() for _ in range(6)]
    A, B, C = [-0.5, -1.5, -2.5], [0.3, -0.7, 1.1], [0.9, 0.2, -0.4]
    h, manual = [0.0] * 3, []
    for t in range(6):
        Abar, Bbar = discretize_zoh(0.4, A, B)
        h = [Abar[n] * h[n] + Bbar[n] * xs[t] for n in range(3)]
        manual.append(sum(C[n] * h[n] for n in range(3)))
    h2, got = [0.0] * 3, []
    for t in range(6):
        h2, y = selective_ssm_step(xs[t], h2, A, B, C, 0.4)
        got.append(y)
    for t in range(6):
        assert got[t] == pytest.approx(manual[t], abs=1e-15)


def test_a_stable_a_decays_the_state():
    A, B, C = [-0.5, -1.5, -2.5], [0.3, -0.7, 1.1], [0.9, 0.2, -0.4]
    h, mags = [1.0, 1.0, 1.0], []
    for _ in range(10):
        h, _ = selective_ssm_step(0.0, h, A, B, C, 0.4)
        mags.append(max(abs(v) for v in h))
    assert all(mags[t + 1] < mags[t] for t in range(len(mags) - 1))
    assert mags[-1] < 0.2


def test_shapes_and_the_skip_connection():
    rng = np.random.default_rng(11)
    X = [[rng.standard_normal() for _ in range(4)] for _ in range(7)]
    r = selective_scan(X, [[-1.0, -2.0]] * 4,
                       [[0.1, 0.2, 0.3, 0.4]] * 2,
                       [[0.5, 0.6, 0.7, 0.8]] * 2,
                       [[0.9, 1.0, 1.1, 1.2]],
                       delta_bias=[0.1, 0.2, 0.3, 0.4])
    assert (r["L"], r["D"], r["N"]) == (7, 4, 2)
    assert len(r["y"]) == 7 and len(r["y"][0]) == 4
    assert len(r["state"]) == 4 and len(r["state"][0]) == 2
    Xv = [[float(v)] for v in (0.0, 3.0, 0.0)]
    base = selective_scan(Xv, [[-1.0]], [[0.4]], [[0.6]], [[1.1]],
                          b_B=[0.2], b_C=[0.3])
    skip = selective_scan(Xv, [[-1.0]], [[0.4]], [[0.6]], [[1.1]],
                          b_B=[0.2], b_C=[0.3], D_skip=[2.0])
    for t in range(len(Xv)):
        assert skip["y"][t][0] == pytest.approx(
            base["y"][t][0] + 2.0 * Xv[t][0], abs=1e-14)


def test_argument_checks():
    Xv = [[float(v)] for v in (0.0, 3.0, 0.0)]
    with pytest.raises(ValueError):
        # s_Delta projects to ONE dimension and is broadcast
        selective_scan(Xv, [[-1.0]], [[0.4]], [[0.6]],
                       [[1.1], [0.5]])
    with pytest.raises(ValueError):
        discretize_zoh(0.5, [-1.0], [1.0], rule="nope")
    with pytest.raises(ValueError):
        discretize_zoh(-0.5, [-1.0], [1.0])
    with pytest.raises(ValueError):
        discretize_zoh(0.5, [-1.0, -2.0], [1.0])
    with pytest.raises(ValueError):
        selective_ssm_step(1.0, [0.0, 0.0], [-1.0], [1.0], [1.0], 0.5)
    with pytest.raises(ValueError):
        selective_ssm_step(1.0, [0.0], [-1.0], [1.0], [1.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        selective_scan([], [[-1.0]], [[0.4]], [[0.6]], [[1.1]])
    with pytest.raises(ValueError):
        selective_scan(Xv, [[-1.0]], [[0.4]], [[0.6]], [[1.1]],
                       delta_bias=[0.1, 0.2])
