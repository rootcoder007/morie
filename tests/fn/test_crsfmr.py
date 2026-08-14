"""crsfmr -- Crossformer. Source: Zhang, Y. & Yan, J. (2023)
"Crossformer: Transformer Utilizing Cross-Dimension Dependency for
Multivariate Time Series Forecasting", ICLR 2023 (no DOI)."""
import pytest

from morie.fn import _array_core as np
from morie.fn.crsfmr import (attention, complexity,
                             cross_dimension_stage, cross_time_stage,
                             dsw_embed, segment_merge,
                             two_stage_attention)

T, D, LSEG = 24, 4, 6


def series(seed=1):
    rng = np.random.default_rng(seed)
    return [[float(rng.normal()) for _ in range(D)] for _ in range(T)]


def test_embedding_shape_is_segments_by_dims_by_model():
    e = dsw_embed(series(), LSEG)
    assert e["shape"] == (T // LSEG, D, LSEG)


def test_the_identity_projection_returns_the_raw_segment():
    X = series()
    e = dsw_embed(X, LSEG)
    assert e["H"][1][2] == [X[LSEG + q][2] for q in range(LSEG)]


def test_segments_partition_the_series():
    X = series()
    e = dsw_embed(X, LSEG)
    flat = [e["H"][i][0][q] for i in range(e["n_seg"])
            for q in range(LSEG)]
    assert flat == [X[t][0] for t in range(T)]


def test_permuting_dimensions_permutes_the_embedding():
    X = series()
    perm = [2, 0, 3, 1]
    Xp = [[X[t][perm[d]] for d in range(D)] for t in range(T)]
    a = dsw_embed(X, LSEG)["H"]
    b = dsw_embed(Xp, LSEG)["H"]
    for i in range(len(a)):
        for d in range(D):
            assert b[i][d] == a[i][perm[d]]


def test_an_indivisible_series_is_refused():
    with pytest.raises(ValueError):
        dsw_embed(series(), 5)


def test_a_zero_segment_length_is_refused():
    with pytest.raises(ValueError):
        dsw_embed(series(), 0)


def test_an_empty_series_is_refused():
    with pytest.raises(ValueError):
        dsw_embed([], 2)


def test_attention_rows_are_distributions():
    KV = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    a = attention([[1.0, 0.0], [0.0, 1.0]], KV, KV)
    for w in a["weights"]:
        assert sum(w) == pytest.approx(1.0, abs=1e-12)
        assert all(v >= 0.0 for v in w)


def test_a_uniquely_aligned_query_concentrates():
    KV = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    w = attention([[50.0, 0.0]], KV, KV)["weights"][0]
    assert w[0] > 0.99


def test_tied_keys_split_the_mass_evenly():
    KV = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    w = attention([[50.0, 0.0]], KV, KV)["weights"][0]
    assert w[0] == pytest.approx(0.5, abs=1e-9)
    assert w[2] == pytest.approx(0.5, abs=1e-9)


def test_mismatched_key_value_lengths_are_refused():
    KV = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        attention([[1.0, 0.0]], KV, KV[:-1])


def test_cross_time_preserves_shape():
    Z = dsw_embed(series(), LSEG)["H"]
    zt = cross_time_stage(Z)
    assert len(zt) == len(Z)
    assert len(zt[0]) == D
    assert len(zt[0][0]) == LSEG


def test_cross_time_is_equivariant_to_dimension_permutation():
    X = series()
    perm = [2, 0, 3, 1]
    Xp = [[X[t][perm[d]] for d in range(D)] for t in range(T)]
    a = cross_time_stage(dsw_embed(X, LSEG)["H"])
    b = cross_time_stage(dsw_embed(Xp, LSEG)["H"])
    for i in range(len(a)):
        for d in range(D):
            for q in range(LSEG):
                assert b[i][d][q] == pytest.approx(a[i][perm[d]][q],
                                                   abs=1e-12)


def test_the_router_changes_the_array():
    Z = cross_time_stage(dsw_embed(series(), LSEG)["H"])
    zd = cross_dimension_stage(Z, n_router=2)
    assert any(abs(zd[i][d][q] - Z[i][d][q]) > 1e-6
               for i in range(len(Z)) for d in range(D)
               for q in range(LSEG))


def test_a_router_below_one_is_refused():
    Z = cross_time_stage(dsw_embed(series(), LSEG)["H"])
    with pytest.raises(ValueError):
        cross_dimension_stage(Z, n_router=0)


def test_a_full_tsa_layer_reports_its_configuration():
    Z = dsw_embed(series(), LSEG)["H"]
    t = two_stage_attention(Z, n_router=2)
    assert t["n_router"] == 2 and t["D"] == D and t["L"] == T // LSEG


def test_router_cost_is_linear_in_the_dimension_count():
    a = complexity(16, 8, 4)["cross_dimension_router"]
    b = complexity(16, 64, 4)["cross_dimension_router"]
    assert b == a * 8


def test_full_cross_dimension_cost_is_quadratic():
    a = complexity(16, 8, 4)["cross_dimension_full"]
    b = complexity(16, 64, 4)["cross_dimension_full"]
    assert b == a * 64


def test_merging_halves_the_segment_count():
    Z = dsw_embed(series(), LSEG)["H"]
    assert len(segment_merge(Z, 2)) == len(Z) // 2


def test_a_merged_vector_is_the_mean_of_its_parts():
    Z = dsw_embed(series(), LSEG)["H"]
    m = segment_merge(Z, 2)
    for d in range(D):
        for q in range(LSEG):
            assert m[0][d][q] == pytest.approx(
                0.5 * (Z[0][d][q] + Z[1][d][q]), abs=1e-12)


def test_an_indivisible_merge_is_refused():
    Z = dsw_embed(series(), LSEG)["H"]
    with pytest.raises(ValueError):
        segment_merge(Z, 3)


def test_a_merge_factor_below_two_is_refused():
    Z = dsw_embed(series(), LSEG)["H"]
    with pytest.raises(ValueError):
        segment_merge(Z, 1)
