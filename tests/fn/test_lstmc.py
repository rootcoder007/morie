"""lstmc: single-step LSTM cell.

Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
*Neural Computation*, 9(8), 1735-1780 -- in the library, verified from the
PDF (page 1 reads "Communicated by Ronald Williams / Long Short-Term Memory /
Sepp Hochreiter").

    f = sigmoid(W_f x + U_f h + b_f)     forget
    i = sigmoid(W_i x + U_i h + b_i)     input
    o = sigmoid(W_o x + U_o h + b_o)     output
    g = tanh(W_g x + U_g h + b_g)        candidate
    c = f * c_prev + i * g
    h = o * tanh(c)
"""

import numpy as np
import pytest

from morie.fn.lstmc import lstm_cell as cell


def _r(H=4, D=3, seed=1, **kw):
    rng = np.random.default_rng(seed)
    return cell(rng.standard_normal(D), hidden_size=H, seed=seed, **kw)


def test_lstmc_gates_are_probabilities_and_candidate_is_bounded():
    """f, i, o pass through a sigmoid so they lie in (0,1); g passes through
    tanh so it lies in (-1,1). Mixing the two up is the classic LSTM bug."""
    r = _r()
    for gate in ("f", "i", "o"):
        v = np.asarray(r[gate])
        assert np.all((v > 0) & (v < 1)), f"gate {gate} outside (0,1)"
    g = np.asarray(r["g"])
    assert np.all((g > -1) & (g < 1))


def test_lstmc_hidden_state_is_bounded_by_the_output_gate():
    """h = o * tanh(c), and both factors are bounded, so |h| < 1."""
    r = _r(H=8, seed=3)
    assert np.all(np.abs(np.asarray(r["h"])) < 1.0)


def test_lstmc_cell_update_follows_the_recurrence():
    """c = f * c_prev + i * g, checked against the returned gates."""
    H, D = 5, 4
    rng = np.random.default_rng(7)
    c_prev = rng.standard_normal(H)
    r = cell(rng.standard_normal(D), c_prev=c_prev, hidden_size=H, seed=7)
    f, i, g, c = (np.asarray(r[k]) for k in ("f", "i", "g", "c"))
    assert c == pytest.approx(f * c_prev + i * g)


def test_lstmc_hidden_follows_the_output_equation():
    r = _r(H=6, seed=11)
    o, c, h = (np.asarray(r[k]) for k in ("o", "c", "h"))
    assert h == pytest.approx(o * np.tanh(c))


def test_lstmc_a_closed_forget_gate_erases_the_past():
    """The constant-error-carousel property the paper exists for: with
    f = 0 the previous cell state contributes nothing, and with f = 1 and
    i = 0 it is carried forward unchanged."""
    H = 4
    c_prev = np.array([5.0, -3.0, 0.5, 2.0])
    # b is laid out [i; f; g; o] (see the module: W is [W_i; W_f; W_g; W_o]),
    # so block 0 is the INPUT gate and block 1 is the FORGET gate.
    b_erase = np.concatenate([np.full(H, -50.0),   # i closed
                              np.full(H, -50.0),   # f closed -> erase
                              np.zeros(H), np.zeros(H)])
    r = cell(np.zeros(3), c_prev=c_prev, hidden_size=H, b=b_erase, seed=1)
    assert np.asarray(r["c"]) == pytest.approx(np.zeros(H), abs=1e-6)


def test_lstmc_an_open_forget_and_closed_input_carries_state_unchanged():
    H = 4
    c_prev = np.array([5.0, -3.0, 0.5, 2.0])
    b_carry = np.concatenate([np.full(H, -50.0),   # i closed -> add nothing
                              np.full(H, 50.0),    # f open   -> keep all
                              np.zeros(H), np.zeros(H)])
    r = cell(np.zeros(3), c_prev=c_prev, hidden_size=H, b=b_carry, seed=1)
    assert np.asarray(r["c"]) == pytest.approx(c_prev, abs=1e-6)


def test_lstmc_is_reproducible_for_a_fixed_seed():
    a = np.asarray(_r(seed=23)["h"])
    b = np.asarray(_r(seed=23)["h"])
    assert a == pytest.approx(b, abs=0.0)


def test_lstmc_shapes_follow_hidden_size():
    r = _r(H=7, D=5, seed=29)
    for k in ("f", "i", "o", "g", "c", "h"):
        assert np.asarray(r[k]).shape == (7,)
