"""Tests for almap.alammar_mean_average_precision."""

from morie.fn import _array_core as np

from morie.fn.almap import alammar_mean_average_precision


def _ap(rel):
    """Independent reference: AP for a single binary relevance list."""
    r = np.asarray(rel, dtype=float).ravel()
    R = float(r.sum())
    idx = np.arange(1, r.size + 1)
    prec = np.cumsum(r) / idx
    return float(np.sum(r * prec) / R)


def test_almap_basic():
    """Test basic functionality with a valid binary relevance input."""
    relevance = [
        [1, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 0],
    ]
    result = alammar_mean_average_precision(relevance)

    # Docstring documents these keys; function returns a RichResult.
    assert isinstance(result, dict)
    for key in (
        "map",
        "average_precisions",
        "precision_at_k",
        "recall_at_k",
        "n_queries",
        "queries_without_relevant",
    ):
        assert key in result

    # n_queries matches the input length.
    assert int(result["n_queries"]) == len(relevance)

    # Per-query APs from the independent formula.
    expected_aps = np.asarray([_ap(q) for q in relevance])
    got_aps = np.asarray(result["average_precisions"], dtype=float)
    assert got_aps.shape == expected_aps.shape
    assert np.allclose(got_aps, expected_aps)

    # MAP is the mean of the per-query APs.
    expected_map = float(expected_aps.mean())
    assert np.isclose(float(result["map"]), expected_map)
    assert np.isclose(float(result["estimate"]), expected_map)

    # precision_at_k and recall_at_k shapes match the input.
    pk = np.asarray(result["precision_at_k"], dtype=float)
    rk = np.asarray(result["recall_at_k"], dtype=float)
    assert pk.shape == (len(relevance),)
    assert rk.shape == (len(relevance),)

    # All three queries have at least one relevant document.
    assert int(result["queries_without_relevant"]) == 0

    # k defaults to None.
    assert result["k"] is None

    # Docstring example reproduces.
    example_res = alammar_mean_average_precision([[1, 0, 1]])
    assert np.isclose(float(example_res["map"]), 5.0 / 6.0)


def test_almap_edge():
    """Test edge cases: a query with no relevant documents is excluded, not zero."""
    relevance = [
        [0, 0, 1, 1],   # 2 relevant
        [0, 0, 0, 0],   # no relevant -> AP undefined, excluded from MAP
        [1, 0, 0, 0],   # 1 relevant, rank 1
    ]
    result = alammar_mean_average_precision(relevance)

    assert isinstance(result, dict)

    # Two of three queries have relevant docs; the third is excluded.
    assert int(result["n_queries"]) == 3
    assert int(result["queries_without_relevant"]) == 1

    # Per-query APs (NaN where there are no relevant docs).
    expected_aps = np.asarray(
        [_ap([0, 0, 1, 1]), float("nan"), _ap([1, 0, 0, 0])],
        dtype=float,
    )
    got_aps = np.asarray(result["average_precisions"], dtype=float)
    assert got_aps.shape == expected_aps.shape
    assert np.allclose(got_aps[:2], expected_aps[:2], equal_nan=True)
    assert np.isnan(got_aps[1])

    # MAP is the mean over the non-NaN queries only.
    good = ~np.isnan(expected_aps)
    expected_map = float(expected_aps[good].mean())
    assert np.isclose(float(result["map"]), expected_map)
    assert np.isclose(float(result["estimate"]), expected_map)

    # precision_at_k is defined for every query (mean of its truncated ranking).
    pk = np.asarray(result["precision_at_k"], dtype=float)
    expected_pk = np.asarray([0.5, 0.0, 0.25], dtype=float)
    assert pk.shape == expected_pk.shape
    assert np.allclose(pk, expected_pk)

    # recall_at_k is NaN for the query with no relevant docs.
    rk = np.asarray(result["recall_at_k"], dtype=float)
    assert rk.shape == (3,)
    assert np.isclose(float(rk[0]), 1.0)
    assert np.isnan(float(rk[1]))
    assert np.isclose(float(rk[2]), 1.0)
