"""Tests for cmlmer.compressed_lmm."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.cmlmer import compressed_lmm


def _to_plain(arr):
    """Convert numpy-like array to a nested list of plain Python floats."""
    # Try .tolist() first (works for the shim when implemented)
    if hasattr(arr, "tolist"):
        out = arr.tolist()
    else:
        out = list(arr)
    # Recursively unwrap
    def _unwrap(x):
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)):
            return [_unwrap(e) for e in x]
        return float(x)
    return _unwrap(out)


def test_cmlmer_basic():
    """Test basic functionality of compressed_lmm."""
    rng = np.random.default_rng(43)

    # Problem size: n individuals must match across y, M, K.
    n = 100
    n_markers = 10

    # Phenotype vector of length n
    y = rng.normal(0, 1, n)

    # Marker matrix of shape (n, n_markers): one column per marker,
    # one row per individual.
    M = rng.normal(0, 1, (n, n_markers))

    # Kinship matrix: symmetric (n, n). Build from A = X X' / p + diag offset,
    # then symmetrise to satisfy the documented symmetry requirement.
    raw = rng.normal(0, 1, (n, n))
    K = raw @ raw.T / n + 0.5 * np.eye(n)
    K = 0.5 * (K + K.T)

    # Number of compressed clusters: a small positive int <= n.
    clusters = 5

    result = compressed_lmm(
        _to_plain(y),
        _to_plain(M),
        _to_plain(K),
        clusters=int(clusters),
    )

    assert isinstance(result, dict)

    # Documented scalar outputs
    for key in ("delta", "sigma2_g", "sigma2_e", "h2",
                "reml_loglik", "n", "n_markers", "p", "n_groups"):
        assert key in result, f"missing key: {key}"

    # Documented per-marker vectors of length n_markers
    n_markers = int(result["n_markers"])
    for key in ("beta", "se", "t", "p_value"):
        assert key in result, f"missing per-marker key: {key}"
        assert len(result[key]) == n_markers

    # Independent check of h2 = sigma2_g / (sigma2_g + sigma2_e)
    sg = float(result["sigma2_g"])
    se = float(result["sigma2_e"])
    expected_h2 = sg / (sg + se)
    assert abs(float(result["h2"]) - expected_h2) < 1e-10

    # Independent check: delta = sigma2_e / sigma2_g  =>  sigma2_e = delta * sigma2_g
    assert abs(se - float(result["delta"]) * sg) < 1e-10

    # P-values lie in [0, 1]
    for pv in result["p_value"]:
        assert 0.0 <= float(pv) <= 1.0


def test_cmlmer_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 50
    n_markers = 4

    y = rng.normal(0, 1, n)
    M = rng.normal(0, 1, (n, n_markers))

    # Symmetric positive-ish kinship
    raw = rng.normal(0, 1, (n, n))
    K = raw @ raw.T / n + 0.5 * np.eye(n)
    K = 0.5 * (K + K.T)

    # Default clusters (None) means no compression: g = n.
    result = compressed_lmm(
        _to_plain(y),
        _to_plain(M),
        _to_plain(K),
        clusters=None,
    )

    assert isinstance(result, dict)
    # With clusters=None the function sets g = n, so n_groups must equal n.
    assert int(result["n_groups"]) == n
    assert len(result["group_sizes"]) == n
    assert all(int(s) == 1 for s in result["group_sizes"])
