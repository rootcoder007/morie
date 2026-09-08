"""spqkv: sparse attention mask (Child et al. 2019, Sparse Transformer).

    sliding window + strided global tokens + optional random connections
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spqkv import sparse_attention as sa


def _allowed(n=32, **kw):
    """Boolean "may attend" view of the mask.

    spqkv returns an ADDITIVE mask (0.0 to keep, -inf to block), the same
    convention as attnq, because it is meant to be added to the attention
    logits before the softmax. It is NOT a boolean array, and casting it to
    int gives INT_MIN for the -inf entries.
    """
    return np.isfinite(np.asarray(sa(np.zeros((n, n)), **kw)["tensor"]))


def test_spqkv_is_sparser_than_dense():
    """The whole point: attention cost drops because most pairs are excluded."""
    r = sa(np.zeros((64, 64)), window=4, stride=8, n_random=0, seed=1)
    assert r["density"] < 1.0
    assert 0.0 < r["density"]


def test_spqkv_every_position_attends_to_itself():
    """A token that cannot see itself has no residual path through
    attention."""
    assert np.all(np.diag(_allowed(32, window=4, stride=8, n_random=0, seed=2)))


def test_spqkv_the_sliding_window_is_always_connected():
    """Positions within `window` of each other must be linked -- that is the
    local half of the Child-2019 pattern."""
    n, w = 32, 3
    m = _allowed(n, window=w, stride=100, n_random=0, seed=3)
    for i in range(n):
        for j in range(max(0, i - w), min(n, i + w + 1)):
            assert m[i, j], f"window pair ({i}, {j}) missing"


def test_spqkv_strided_global_tokens_are_reachable():
    """The global half: every `stride`-th column is attended broadly, which
    is what keeps long-range information flowing at all."""
    n, s = 48, 8
    m = _allowed(n, window=1, stride=s, n_random=0, seed=5)
    global_cols = m[:, ::s].mean()
    other_cols = m[:, [c for c in range(n) if c % s]].mean()
    assert global_cols > other_cols


def test_spqkv_density_falls_as_the_window_narrows():
    d = [sa(np.zeros((64, 64)), window=w, stride=16, n_random=0, seed=7)["density"]
         for w in (16, 8, 4, 2)]
    assert d == sorted(d, reverse=True)


def test_spqkv_random_connections_add_density_and_respect_the_seed():
    a = sa(np.zeros((48, 48)), window=2, stride=16, n_random=0, seed=11)
    b = sa(np.zeros((48, 48)), window=2, stride=16, n_random=5, seed=11)
    assert b["density"] > a["density"]
    c = sa(np.zeros((48, 48)), window=2, stride=16, n_random=5, seed=11)
    assert np.array_equal(np.asarray(b["tensor"]), np.asarray(c["tensor"]))


def test_spqkv_mask_is_additive_zero_or_minus_inf():
    """0.0 keeps, -inf blocks -- ready to add to the logits. Anything else
    would corrupt the softmax rather than mask it."""
    r = sa(np.zeros((24, 24)), window=3, stride=6, n_random=2, seed=13)
    m = np.asarray(r["tensor"])
    assert m.shape == (24, 24)
    assert set(np.unique(m).tolist()) <= {0.0, -np.inf}


def test_spqkv_masked_logits_get_zero_attention_weight():
    """End to end: feeding this mask through a softmax must zero exactly the
    blocked pairs, which is the only thing the mask is for."""
    n = 16
    m = np.asarray(sa(np.zeros((n, n)), window=2, stride=8, n_random=0, seed=17)["tensor"])
    rng = np.random.default_rng(19)
    logits = rng.standard_normal((n, n)) + m
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    w = e / e.sum(axis=-1, keepdims=True)
    assert np.allclose(w[~np.isfinite(m)], 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)
