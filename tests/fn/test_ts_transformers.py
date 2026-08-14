"""informer / infmer / patchT.

Sources: Zhou, H. et al. (2021) AAAI 35(12), 11106-11115,
arXiv:2012.07436; Nie, Y., Nguyen, N. H., Sinthong, P. &
Kalagnanam, J. (2023) ICLR 2023, arXiv:2211.14730."""
import math

import pytest

from morie.fn import _array_core as np
from morie.fn import infmer
from morie.fn.informer import (complexity, full_attention,
                               kl_from_uniform, probsparse_attention,
                               select_queries, sparsity_measure)
from morie.fn.patchT import (attention_cost,
                             channel_independent_tokens,
                             channel_mixed_tokens, instance_norm,
                             patchify, patchtst_encode)

K0 = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]


def qkv(LQ=32, LK=32, D=4, seed=1):
    rng = np.random.default_rng(seed)
    Q = [[float(rng.normal()) for _ in range(D)] for _ in range(LQ)]
    K = [[float(rng.normal()) for _ in range(D)] for _ in range(LK)]
    V = [[float(rng.normal()) for _ in range(D)] for _ in range(LK)]
    return Q, K, V


def test_the_measure_equals_log_L_at_uniform_attention():
    assert sparsity_measure([0.0, 0.0], K0) == pytest.approx(
        math.log(4), abs=1e-12)


def test_the_kl_is_zero_at_uniform_attention():
    assert kl_from_uniform([0.0, 0.0], K0) == pytest.approx(
        0.0, abs=1e-12)


def test_the_kl_is_positive_for_a_peaked_query():
    assert kl_from_uniform([30.0, 0.0], K0) > 1.0


def test_the_measure_never_falls_below_log_L():
    rng = np.random.default_rng(3)
    for _ in range(30):
        q = [float(rng.normal()), float(rng.normal())]
        assert sparsity_measure(q, K0) >= math.log(4) - 1e-12


def test_the_maxmean_bound_is_zero_at_uniform():
    assert sparsity_measure([0.0, 0.0], K0,
                            measure="maxmean") == pytest.approx(
        0.0, abs=1e-12)


def test_u_is_the_stated_function_of_the_query_count():
    Q, K, _ = qkv(LQ=64)
    s = select_queries(Q, K, factor=5)
    assert s["u"] == max(1, min(64, int(5 * math.log(64))))
    assert len(s["top"]) == s["u"]


def test_a_full_budget_reproduces_full_attention_exactly():
    Q, K, V = qkv()
    ps = probsparse_attention(Q, K, V, factor=10000)
    fa = full_attention(Q, K, V)
    for i in range(len(Q)):
        for a in range(len(V[0])):
            assert ps["output"][i][a] == pytest.approx(fa[i][a],
                                                       abs=1e-12)


def test_selected_queries_are_computed_exactly():
    Q, K, V = qkv()
    ps = probsparse_attention(Q, K, V, factor=3)
    fa = full_attention(Q, K, V)
    for i in ps["selected"]:
        for a in range(len(V[0])):
            assert ps["output"][i][a] == pytest.approx(fa[i][a],
                                                       abs=1e-12)


def test_unselected_queries_take_the_value_mean():
    Q, K, V = qkv()
    ps = probsparse_attention(Q, K, V, factor=1)
    vbar = [sum(V[j][a] for j in range(len(V))) / len(V)
            for a in range(len(V[0]))]
    unsel = [i for i in range(len(Q)) if i not in set(ps["selected"])]
    assert unsel
    for a in range(len(V[0])):
        assert ps["output"][unsel[0]][a] == pytest.approx(vbar[a],
                                                          abs=1e-12)


def test_probsparse_is_cheaper_and_more_so_as_L_grows():
    a = complexity(512, 512)
    b = complexity(4096, 4096)
    assert a["probsparse"] < a["full"]
    assert b["ratio"] > a["ratio"]


def test_infmer_is_the_same_implementation():
    Q, K, V = qkv()
    a = probsparse_attention(Q, K, V, factor=3)["output"]
    b = infmer.probsparse_attention(Q, K, V, factor=3)["output"]
    assert a == b


def test_an_empty_key_set_is_refused():
    with pytest.raises(ValueError):
        sparsity_measure([1.0], [])


def test_a_dimension_mismatch_is_refused():
    with pytest.raises(ValueError):
        sparsity_measure([1.0, 2.0, 3.0], K0)


def test_an_unknown_measure_is_refused():
    with pytest.raises(ValueError):
        sparsity_measure([0.0, 0.0], K0, measure="entropy")


# ------------------------------------------------------------ patchT
def series(L=96, D=4, seed=2):
    rng = np.random.default_rng(seed)
    return [[float(rng.normal()) for _ in range(D)] for _ in range(L)]


def test_the_patch_count_is_the_stated_formula():
    p = patchify([float(i) for i in range(96)], 16, 8)
    assert p["n_patches"] == (96 - 16) // 8 + 1


def test_every_patch_has_the_stated_length():
    p = patchify([float(i) for i in range(96)], 16, 8)
    assert all(len(q) == 16 for q in p["patches"])


def test_disjoint_patches_cover_the_series():
    p = patchify([float(i) for i in range(96)], 16, 16)
    assert sum(len(q) for q in p["patches"]) == 96


def test_channel_independence_is_equivariant():
    X = series()
    perm = [2, 0, 3, 1]
    Xp = [[X[t][perm[d]] for d in range(4)] for t in range(len(X))]
    a = channel_independent_tokens(X, 16, 8)["tokens"]
    b = channel_independent_tokens(Xp, 16, 8)["tokens"]
    for d in range(4):
        assert b[d] == a[perm[d]]


def test_channel_mixing_is_permutation_invariant():
    X = series()
    perm = [2, 0, 3, 1]
    Xp = [[X[t][perm[d]] for d in range(4)] for t in range(len(X))]
    a = channel_mixed_tokens(X, 16, 8)["tokens"]
    b = channel_mixed_tokens(Xp, 16, 8)["tokens"]
    for i in range(len(a)):
        for j in range(len(a[0])):
            assert b[i][j] == pytest.approx(a[i][j], abs=1e-12)


def test_channel_independence_makes_D_times_more_tokens():
    X = series()
    ci = channel_independent_tokens(X, 16, 8)["n_tokens_total"]
    cm = channel_mixed_tokens(X, 16, 8)["n_tokens_total"]
    assert ci == cm * 4


def test_patching_shrinks_the_attention_map():
    c = attention_cost(96, 16, 8, 4)
    n = (96 - 16) // 8 + 1
    assert c["reduction"] == pytest.approx((96 * 96) / (n * n),
                                           abs=1e-9)


def test_a_larger_stride_shrinks_it_further():
    a = attention_cost(96, 16, 4)["patched"]
    b = attention_cost(96, 16, 16)["patched"]
    assert b < a


def test_instance_norm_standardises():
    v = [float(i) * 0.3 for i in range(50)]
    z = instance_norm(v)["normalised"]
    m = sum(z) / len(z)
    sd = math.sqrt(sum((q - m) ** 2 for q in z) / (len(z) - 1))
    assert m == pytest.approx(0.0, abs=1e-12)
    assert sd == pytest.approx(1.0, abs=1e-12)


def test_a_constant_series_is_flagged_degenerate():
    assert instance_norm([2.0] * 10)["degenerate"]


def test_the_encoder_returns_one_stream_per_channel():
    e = patchtst_encode(series(), 16, 8)
    assert len(e["tokens"]) == 4
    assert len(e["norm_stats"]) == 4


def test_a_patch_longer_than_the_series_is_refused():
    with pytest.raises(ValueError):
        patchify([1.0, 2.0, 3.0], 10)


def test_a_zero_patch_length_is_refused():
    with pytest.raises(ValueError):
        patchify([1.0] * 20, 0)


def test_a_zero_stride_is_refused():
    with pytest.raises(ValueError):
        patchify([1.0] * 20, 4, 0)


def test_an_empty_series_is_refused():
    with pytest.raises(ValueError):
        channel_independent_tokens([], 4)


def test_normalising_one_point_is_refused():
    with pytest.raises(ValueError):
        instance_norm([1.0])
