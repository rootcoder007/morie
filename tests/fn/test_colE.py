"""Tests for colE.cold_start_user."""

import math

from morie.fn import _array_core as np

from morie.fn.colE import cold_start_user


def _make_R(rng, n_users, n_items, obs_prob=0.7):
    """Build a sparse rating matrix where 0 means unobserved."""
    R = []
    for i in range(n_users):
        row = []
        for j in range(n_items):
            if rng.uniform(0, 1) < obs_prob:
                row.append(rng.uniform(1, 5))
            else:
                row.append(0.0)
        R.append(row)
    return R


def test_colE_basic():
    """Test basic functionality with the popular fallback."""
    rng = np.random.default_rng(42)
    n_users, n_items = 40, 10
    R = _make_R(rng, n_users, n_items, obs_prob=0.7)

    user = 0
    topn = 3
    result = cold_start_user(user, "popular", R=R, topn=topn)

    assert isinstance(result, dict)
    for key in ("estimate", "is_cold", "n_rated", "scores",
                "recommended", "mode", "n_users", "n_items"):
        assert key in result

    assert result["n_users"] == n_users
    assert result["n_items"] == n_items
    assert result["mode"] == "popular"
    assert math.isfinite(result["estimate"])
    assert 0 < len(result["recommended"]) <= topn
    assert len(result["scores"]) == n_items
    assert isinstance(result["is_cold"], int)


def test_colE_edge():
    """Test edge case with a cold-start user (fewer than min_ratings)."""
    rng = np.random.default_rng(123)
    n_users, n_items = 40, 10
    R = _make_R(rng, n_users, n_items, obs_prob=0.7)
    for j in range(n_items):
        R[5][j] = 0.0

    user = 5
    result = cold_start_user(user, "popular", R=R, min_ratings=3)

    assert isinstance(result, dict)
    assert result["is_cold"] == 1
    assert result["n_rated"] == 0
    assert result["n_users"] == n_users
    assert result["n_items"] == n_items
