"""Tests for gan_an.gan_anomaly."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gan_an import gan_anomaly


def _make_identity_generator(out_dim, z_dim):
    """Trivial generator: maps z -> z projected to out_dim (first z_dim kept, rest zeros)."""
    def gen(z):
        z = np.asarray(z, dtype=float).ravel().tolist()
        out = [float(z[i]) if i < len(z) else 0.0 for i in range(out_dim)]
        return out
    return gen


def _make_feature_fn(out_dim):
    """Trivial feature_fn: returns the first out_dim entries of x."""
    def feat(x):
        x = np.asarray(x, dtype=float).ravel().tolist()
        return [float(x[i]) if i < len(x) else 0.0 for i in range(out_dim)]
    return feat


def test_gan_an_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, (8,)).tolist()
    z_dim = 5
    out_dim = 8
    feat_dim = 4
    generator = _make_identity_generator(out_dim, z_dim)
    feature_fn = _make_feature_fn(feat_dim)

    result = gan_anomaly(X, generator, feature_fn, z_dim,
                         steps=10, lr=0.1, lam=0.5, seed=0,
                         h=1e-3, step_decay=0.05)

    assert isinstance(result, dict)
    # Documented keys present
    for key in ("estimate", "score", "z", "reconstruction",
                "loss_history", "residual", "discrimination",
                "final_step", "method", "note"):
        assert key in result, f"missing key: {key}"

    # Shapes / types per docstring
    assert len(result["z"]) == z_dim
    assert len(result["reconstruction"]) == out_dim
    assert isinstance(result["loss_history"], list)
    assert len(result["loss_history"]) == 10
    assert isinstance(result["estimate"], float)
    assert isinstance(result["score"], float)
    assert isinstance(result["residual"], float)
    assert isinstance(result["discrimination"], float)
    assert isinstance(result["final_step"], float)

    # final_step = lr / (1 + (steps - 1) * step_decay) per formula
    steps = 10
    lr = 0.1
    step_decay = 0.05
    expected_final_step = lr / (1.0 + (steps - 1) * step_decay)
    assert abs(result["final_step"] - expected_final_step) < 1e-12


def test_gan_an_edge():
    """Test edge cases: minimal steps, step_decay=0 (fixed step)."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, (4,)).tolist()
    z_dim = 3
    out_dim = 4
    feat_dim = 2
    generator = _make_identity_generator(out_dim, z_dim)
    feature_fn = _make_feature_fn(feat_dim)

    # step_decay=0 -> fixed step (per docstring)
    result = gan_anomaly(X, generator, feature_fn, z_dim,
                         steps=3, lr=0.01, lam=0.2, seed=7,
                         h=1e-3, step_decay=0.0)
    assert isinstance(result, dict)
    assert "score" in result
    assert len(result["z"]) == z_dim
    assert len(result["loss_history"]) == 3
    # With step_decay=0, final_step == lr
    assert abs(result["final_step"] - 0.01) < 1e-12
