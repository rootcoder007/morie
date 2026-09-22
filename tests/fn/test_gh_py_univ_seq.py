"""Tests for gh_py_univ_seq.ghosal_py_universal_sequence."""

from morie.fn import _array_core as np

from morie.fn.gh_py_univ_seq import ghosal_py_universal_sequence


def test_gh_py_univ_seq_basic():
    """Test basic functionality with the documented (d, theta, k_list) signature."""
    d = 0.4
    theta = 2.0
    k_list = (1, 5, 20)

    result = ghosal_py_universal_sequence(d=d, theta=theta, k_list=k_list)

    # Documented keys
    assert "estimate" in result
    assert "mean_by_k" in result
    assert "decreasing" in result
    assert "method" in result

    # Independent recomputation of E V_k = (1 - d) / (1 - d + theta + k * d)
    expected_means = [
        (1.0 - d) / (1.0 - d + theta + k * d) for k in k_list
    ]

    # `estimate` is documented as the first mean
    assert result["estimate"] == expected_means[0]

    # `mean_by_k` carries the full decreasing sequence
    assert list(result["mean_by_k"]) == expected_means

    # Finite numeric sanity (avoids bare-number literals)
    arr = np.asarray(result["mean_by_k"], dtype=float)
    assert np.all(np.isfinite(arr))

    # Documented monotonicity: E V_k is strictly decreasing in k
    assert result["decreasing"] is True


def test_gh_py_univ_seq_edge():
    """Test edge case: a single-element k_list yields estimate == mean_by_k[0]."""
    d = 0.4
    theta = 2.0
    k_list = (7,)

    result = ghosal_py_universal_sequence(d=d, theta=theta, k_list=k_list)

    expected = (1.0 - d) / (1.0 - d + theta + k_list[0] * d)

    assert result["estimate"] == expected
    assert list(result["mean_by_k"]) == [expected]
    assert result["decreasing"] is True
