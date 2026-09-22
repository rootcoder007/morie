"""Tests for ffmFM.field_aware_fm."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.ffmFM import field_aware_fm


def _to_rows(X, fields):
    """Convert a dense (n_samples, n_features) matrix into a list of
    sparse-like row representations compatible with field_aware_fm.

    Each row is a list of (feature_index, value) pairs for entries
    whose absolute value exceeds 1e-12. ``fields`` is a length-1 list
    repeated to cover every feature index (a single "field" is fine
    for a smoke test; the field-aware math is exercised regardless).
    """
    n_samples, n_features = X.shape
    rows = []
    for i in range(n_samples):
        row = [(j, float(X[i, j])) for j in range(n_features)
               if abs(float(X[i, j])) > 1e-12]
        rows.append(row)
    return rows


def _to_labels(y):
    """Map continuous labels into {-1, +1} as required by the
    documented logistic-loss formulation."""
    out = []
    for v in y:
        out.append(1.0 if float(v) >= 0.0 else -1.0)
    return out


def test_ffmFM_basic():
    """Test basic functionality on a tiny, deterministic dataset.

    The function implements the field-aware factorization machine
    trained with AdaGrad on the logistic loss.  Its signature is::

        field_aware_fm(rows, labels, fields, n_features,
                       n_fields, k_dim=4, eta=0.1, lam=2e-5,
                       epochs=10, seed=0)

    where ``rows`` is an iterable of sparse rows (each a list of
    ``(feature_index, value)`` pairs), ``labels`` are ±1 targets,
    and ``fields`` is a length-``n_features`` array giving the field
    of every feature.  The test builds small, well-shaped inputs and
    checks the documented keys of the returned ``RichResult.payload``.
    """
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)

    n_samples = 20
    n_features = 5
    n_fields = 2

    # Dense design matrix; converted below into sparse rows.
    X = rng_X.normal(0.0, 1.0, (n_samples, n_features))
    y = rng_y.normal(0.0, 1.0, n_samples)

    rows = _to_rows(X, fields=None)
    labels = _to_labels(y)

    # A length-n_features assignment of features to fields.  Round-robin
    # assignment is deterministic and satisfies the documented shape.
    fields = [i % n_fields for i in range(n_features)]

    result = field_aware_fm(
        rows, labels, fields,
        n_features=n_features, n_fields=n_fields,
        k_dim=4, eta=0.1, lam=2e-5, epochs=5, seed=0,
    )

    # Documented return value is a RichResult; its .payload is a dict.
    assert hasattr(result, "payload")
    payload = result.payload
    assert isinstance(payload, dict)

    # The payload exposes both spellings of the weight tensor so that
    # callers can pick whichever they prefer.
    assert "estimate" in payload
    assert "W" in payload
    assert payload["estimate"] is payload["W"]

    # The logistic-loss curve must have exactly ``epochs`` entries and
    # be monotonically non-increasing for a well-posed dataset.
    assert "loss_history" in payload
    hist = payload["loss_history"]
    assert len(hist) == 5
    for a, b in zip(hist, hist[1:]):
        assert b <= a + 1e-9

    # The last entry of the history must equal ``final_loss``.
    assert "final_loss" in payload
    assert payload["final_loss"] == hist[-1]

    # Documented bookkeeping keys.
    assert payload["k"] == 4
    assert payload["n_parameters"] == n_features * n_fields * 4
    assert payload["n_parameters_fm"] == n_features * 4
    assert "FFM" in payload["method"]


def test_ffmFM_edge():
    """Edge case: a single field makes the model collapse to a plain
    FM, but the call must still succeed and return the documented
    payload keys with the right shapes."""
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)

    n_samples = 12
    n_features = 4
    n_fields = 1  # every feature lives in the single field

    X = rng_X.normal(0.0, 1.0, (n_samples, n_features))
    y = rng_y.normal(0.0, 1.0, n_samples)

    rows = _to_rows(X, fields=None)
    labels = _to_labels(y)
    fields = [0] * n_features  # all features in field 0

    result = field_aware_fm(
        rows, labels, fields,
        n_features=n_features, n_fields=n_fields,
        k_dim=3, eta=0.05, lam=1e-4, epochs=3, seed=7,
    )

    payload = result.payload
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert "W" in payload
    assert payload["estimate"] is payload["W"]
    assert "loss_history" in payload
    assert len(payload["loss_history"]) == 3
    assert "final_loss" in payload
    assert payload["final_loss"] == payload["loss_history"][-1]
    assert payload["k"] == 3
    # n_parameters == n_features * n_fields * k
    assert payload["n_parameters"] == n_features * n_fields * 3
    assert payload["n_parameters_fm"] == n_features * 3
