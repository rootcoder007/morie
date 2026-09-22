"""Tests for causivla.causal_iv_late."""

from morie.fn import _array_core as np

from morie.fn.causivla import causal_iv_late


def _binary_z(seed, n):
    """Build a binary {0,1} instrument from a continuous draw."""
    return (np.random.default_rng(seed).normal(0, 1, n) > 0).astype(float)


def _binary_d_from_z(z, seed):
    """Build a binary treatment D that depends positively on Z (first stage)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, z.size)
    # probability of D=1 increases with Z, so E[D|Z=1] > E[D|Z=0]
    p = 1.0 / (1.0 + np.exp(-(base + 1.0 * z)))
    return (p > 0.5).astype(float)


def test_causivla_basic():
    """Test basic functionality against the Wald ratio formula."""
    n = 1000
    rng = np.random.default_rng(43)

    Z = _binary_z(43, n)
    D = _binary_d_from_z(Z, seed=42)
    y = np.zeros(n)
    # y = true_LATE * D + noise  =>  E[Y|Z=z] = true_LATE * E[D|Z=z]
    true_LATE = 1.5
    eps = rng.normal(0, 1, n)
    y = true_LATE * D + eps

    result = causal_iv_late(y, D, Z)

    # The function returns a dict-like RichResult; assert on documented keys.
    assert isinstance(result, dict)
    for key in ("late", "se", "first_stage", "reduced_form",
                "complier_share", "n_z1", "n_z0", "n", "method"):
        assert key in result, f"missing documented key: {key}"

    # Reconstruct the Wald ratio by hand from the same inputs and compare.
    z1 = Z == 1
    z0 = Z == 0
    rf_hand = float(y[z1].mean() - y[z0].mean())
    fs_hand = float(D[z1].mean() - D[z0].mean())
    late_hand = rf_hand / fs_hand

    assert result["first_stage"] == fs_hand
    assert result["reduced_form"] == rf_hand
    assert result["late"] == late_hand
    assert result["complier_share"] == fs_hand  # docstring: P(D_1 > D_0)
    assert result["n"] == n
    assert result["n_z1"] == int(z1.sum())
    assert result["n_z0"] == int(z0.sum())
    assert result["weak_first_stage"] == bool(abs(fs_hand) < 0.05)
    assert result["method"].startswith("Imbens-Angrist")


def test_causivla_edge():
    """Test edge case: a small but valid sample still returns documented keys."""
    rng = np.random.default_rng(43)
    n = 50
    Z = _binary_z(43, n)
    D = _binary_d_from_z(Z, seed=42)
    eps = rng.normal(0, 1, n)
    y = 0.5 * D + eps

    result = causal_iv_late(y, D, Z)

    assert isinstance(result, dict)
    for key in ("late", "first_stage", "reduced_form", "n_z1", "n_z0"):
        assert key in result, f"missing documented key: {key}"
