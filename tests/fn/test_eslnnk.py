"""Tests for eslnnk.esl_nadaraya_watson."""

from morie.fn import _array_core as np

from morie.fn.eslnnk import esl_nadaraya_watson


def _ep_weights(x0, x_data, y_data, lam):
    """Independent Epanechnikov Nadaraya-Watson implementation for cross-check."""
    num = 0.0
    den = 0.0
    for xi, yi in zip(x_data, y_data):
        t = abs(x0 - xi) / lam
        if t < 1.0:
            k = 0.75 * (1.0 - t * t)
        else:
            k = 0.0
        num += k * yi
        den += k
    return num / den if den > 0 else float("nan")


def test_eslnnk_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    x_data = list(rng.normal(0.0, 1.0, 100))
    y_data = list(rng.normal(0.0, 1.0, 100))
    x0 = float(rng.normal(0.0, 1.0))
    lambda_ = 0.5

    result = esl_nadaraya_watson(x0, x_data, y_data, lambda_)

    assert isinstance(result, dict)
    assert "estimate" in result
    expected = _ep_weights(x0, x_data, y_data, lambda_)
    assert abs(result["estimate"] - expected) < 1e-12

    # Documented result keys
    for key in ("estimate", "values", "effective_n", "n_in_window",
                "lambda", "kernel", "n", "method"):
        assert key in result

    assert result["lambda"] == lambda_
    assert result["kernel"] == "epanechnikov"
    assert result["n"] == 100
    assert len(result["values"]) == 1
    assert len(result["effective_n"]) == 1
    assert len(result["n_in_window"]) == 1


def test_eslnnk_edge():
    """Test edge cases: constant response and out-of-window query."""
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [5.0, 5.0, 5.0, 5.0]

    # Constant response -> estimate equals that constant.
    r1 = esl_nadaraya_watson(1.5, xs, ys, 1.0)
    assert r1["estimate"] == 5.0
    assert r1["n_in_window"] == [2]

    # Linear response y = x -> estimate equals x0 at this bandwidth.
    ys_lin = [0.0, 1.0, 2.0, 3.0]
    r2 = esl_nadaraya_watson(1.5, xs, ys_lin, 1.0)
    assert round(r2["estimate"], 12) == 1.5

    # Query point outside every compact window -> nan.
    r3 = esl_nadaraya_watson(6.0, xs, ys, 1.0)
    assert r3["estimate"] != r3["estimate"]  # nan check
    assert r3["n_in_window"] == [0]
    assert r3["effective_n"] == [0.0]
