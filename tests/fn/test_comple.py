"""Tests for complE.complex."""

from morie.fn import _array_core as np

from morie.fn.comple import complex


def test_comple_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_entities = 5
    n_relations = 3
    dim = 4
    triples = rng.integers(0, n_entities, size=(10, 3))
    triples[:, 1] = rng.integers(0, n_relations, size=10)
    re_e = rng.normal(0, 1, (n_entities, dim))
    im_e = rng.normal(0, 1, (n_entities, dim))
    re_r = rng.normal(0, 1, (n_relations, dim))
    im_r = rng.normal(0, 1, (n_relations, dim))
    result = complex(triples, dim, re_e=re_e, im_e=im_e, re_r=re_r, im_r=im_r)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scores" in result
    assert "m" in result
    assert "dim" in result
    assert result["m"] == 10
    assert result["dim"] == dim

    # Independently compute scores using the documented formula.
    expected_scores = []
    for h, r, t in triples:
        s = 0.0
        for k in range(dim):
            s += (re_e[h, k] * re_r[r, k] * re_e[t, k]
                  + re_e[h, k] * im_r[r, k] * im_e[t, k]
                  + im_e[h, k] * re_r[r, k] * im_e[t, k]
                  - im_e[h, k] * im_r[r, k] * re_e[t, k])
        expected_scores.append(s)
    expected_estimate = sum(expected_scores) / len(expected_scores)

    assert len(result["scores"]) == len(expected_scores)
    for got, exp in zip(result["scores"], expected_scores):
        assert abs(got - exp) < 1e-9
    assert abs(result["estimate"] - expected_estimate) < 1e-9


def test_comple_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_entities = 4
    n_relations = 2
    dim = 3
    triples = rng.integers(0, n_entities, size=(5, 3))
    triples[:, 1] = rng.integers(0, n_relations, size=5)
    re_e = rng.normal(0, 1, (n_entities, dim))
    im_e = rng.normal(0, 1, (n_entities, dim))
    re_r = rng.normal(0, 1, (n_relations, dim))
    im_r = rng.normal(0, 1, (n_relations, dim))
    result = complex(triples, dim, re_e=re_e, im_e=im_e, re_r=re_r, im_r=im_r)
    assert isinstance(result, dict)
    assert "scores" in result
    assert "estimate" in result
    assert result["m"] == 5
    assert result["dim"] == dim

    # Single triple sanity check.
    single = np.array([[0, 0, 1]])
    single_result = complex(single, dim, re_e=re_e, im_e=im_e, re_r=re_r, im_r=im_r)
    h, r, t = 0, 0, 1
    expected = 0.0
    for k in range(dim):
        expected += (re_e[h, k] * re_r[r, k] * re_e[t, k]
                     + re_e[h, k] * im_r[r, k] * im_e[t, k]
                     + im_e[h, k] * re_r[r, k] * im_e[t, k]
                     - im_e[h, k] * im_r[r, k] * re_e[t, k])
    assert abs(single_result["scores"][0] - expected) < 1e-9
