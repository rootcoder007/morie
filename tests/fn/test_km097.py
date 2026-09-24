"""Tests for km097.kamath_ch6_ear_entropy_reg."""

import math

from morie.fn import _array_core as np
from morie.fn.km097 import kamath_ch6_ear_entropy_reg


def _make_attention_layer(rng, n_rows, n_ctx):
    """Build a valid attention layer whose rows are probability distributions."""
    rows = []
    for _ in range(n_rows):
        raw = [float(x) for x in rng.uniform(0.0, 1.0, n_ctx)]
        total = sum(raw)
        rows.append([v / total for v in raw])
    return rows


def test_km097_basic():
    """Test basic functionality with random, well-formed attention matrices."""
    rng = np.random.default_rng(42)
    n_layers = 2
    A = [_make_attention_layer(rng, n_rows=4, n_ctx=5)
         for _ in range(n_layers)]

    result = kamath_ch6_ear_entropy_reg(A, lam=0.5)

    assert isinstance(result, dict)
    for key in ("estimate", "per_layer_entropy", "total_entropy",
                "lam", "n", "method"):
        assert key in result

    assert result["n"] == n_layers
    assert result["lam"] == 0.5
    assert len(result["per_layer_entropy"]) == n_layers

    for h in result["per_layer_entropy"]:
        assert math.isfinite(h)
        assert 0.0 <= h <= math.log(5) + 1e-9

    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["total_entropy"])
    # estimate = -lam * total_entropy, hence non-positive for lam >= 0
    assert result["estimate"] <= 1e-12
    assert math.isclose(result["estimate"],
                        -0.5 * result["total_entropy"],
                        rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["total_entropy"],
                        sum(result["per_layer_entropy"]),
                        rel_tol=1e-12, abs_tol=1e-12)


def test_km097_edge():
    """Test edge cases matching the docstring examples."""
    # Uniform single row: row-entropy = log(2), estimate = -lam * log(2).
    out = kamath_ch6_ear_entropy_reg([[[0.5, 0.5]]], lam=1.0)
    assert isinstance(out, dict)
    assert out["n"] == 1
    assert len(out["per_layer_entropy"]) == 1
    assert math.isclose(out["per_layer_entropy"][0], math.log(2.0),
                        rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out["total_entropy"], math.log(2.0),
                        rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out["estimate"], -math.log(2.0),
                        rel_tol=1e-12, abs_tol=1e-12)

    # Peaked single row: row-entropy = 0, estimate = 0 (with sign).
    out2 = kamath_ch6_ear_entropy_reg([[[1.0, 0.0]]])
    assert isinstance(out2, dict)
    assert out2["n"] == 1
    assert math.isclose(out2["per_layer_entropy"][0], 0.0, abs_tol=1e-12)
    assert math.isclose(out2["total_entropy"], 0.0, abs_tol=1e-12)
    assert math.isclose(out2["estimate"], 0.0, abs_tol=1e-12)
