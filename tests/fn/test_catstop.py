"""Tests for catstop.cat_stopping_rule."""

from morie.fn import _array_core as np

from morie.fn.catstop import cat_stopping_rule


def _irf_4pl(theta, a, b, c, d):
    """4PL IRF: P = c + (d - c) / (1 + exp(-(a*(theta - b))))."""
    return c + (d - c) / (1.0 + np.exp(-(a * (theta - b))))


def _item_info_4pl(theta, a, b, c, d):
    """Information for one 4PL item at theta."""
    e = np.exp(a * (theta - b))
    P = c + (d - c) * e / (1.0 + e)
    Q = 1.0 - P
    dP_dtheta = a * (d - c) * e / (1.0 + e) ** 2
    return (dP_dtheta ** 2) / (P * Q), P


def test_catstop_basic():
    """Test basic functionality with 2PL items (c=0, d=1)."""
    rng = np.random.default_rng(42)
    a = rng.uniform(0.5, 2.0, 10)
    b = rng.uniform(-2.0, 2.0, 10)
    items = [(float(ai), float(bi), 0.0, 1.0) for ai, bi in zip(a, b)]
    theta = 0.5
    se_target = 0.3

    result = cat_stopping_rule(items, theta, se_target)

    # Result is a mapping (RichResult, dict-like)
    assert isinstance(result, dict)

    # Independently compute total information and SE per catR Eqs. 3-4
    infos = [
        _item_info_4pl(theta, ai, bi, 0.0, 1.0)[0] for ai, bi in zip(a, b)
    ]
    total_info = float(sum(infos))
    expected_se = 1.0 / np.sqrt(total_info)

    # Documented return keys
    for key in ("stop", "se", "information", "item_information", "n_items"):
        assert key in result, f"missing key: {key}"

    assert isinstance(result["stop"], bool)
    assert result["se"] == expected_se
    assert result["information"] == total_info
    assert list(result["item_information"]) == infos
    assert result["n_items"] == len(items)
    assert result["stop"] == (expected_se <= se_target)


def test_catstop_edge():
    """Test edge case: many easy items should reach a low SE and stop."""
    rng = np.random.default_rng(42)
    n = 50
    a = rng.uniform(1.0, 2.0, n)
    b = np.linspace(-2.0, 2.0, n)  # items spread across theta=0
    items = [(float(ai), float(bi), 0.0, 1.0) for ai, bi in zip(a, b)]
    theta = 0.0
    se_target = 0.5  # generous target; high-info item bank should stop

    result = cat_stopping_rule(items, theta, se_target)
    assert isinstance(result, dict)

    infos = [
        _item_info_4pl(theta, ai, bi, 0.0, 1.0)[0] for ai, bi in zip(a, b)
    ]
    total_info = float(sum(infos))
    expected_se = 1.0 / np.sqrt(total_info)

    assert result["se"] == expected_se
    assert result["n_items"] == n
    # With a well-targeted bank, SE should be well under 0.5 -> stop=True
    assert result["stop"] is True
