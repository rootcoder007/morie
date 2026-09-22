"""Tests for causmtchcm.causal_caliper_matching."""

from morie.fn import _array_core as np

from morie.fn.causmtchcm import causal_caliper_matching


def test_causmtchcm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ps = rng.beta(2, 2, 100)
    treat = (rng.random(100) < ps).astype(float)
    result = causal_caliper_matching(ps, treat)

    assert "matches" in result
    assert "distances" in result
    assert "n_unmatched" in result
    assert "caliper_used" in result
    assert "match_rate" in result
    assert "estimand" in result

    # Default caliper is 0.2 sd of the logit of the propensity score
    logit = np.log(ps / (1.0 - ps))
    expected_caliper = float(0.2 * np.std(logit, ddof=1))
    assert result["caliper_used"] == expected_caliper

    # All returned matches must satisfy the caliper constraint on the
    # matching scale (logit by default).
    n_treated = int((treat == 1).sum())
    assert result["matches"].shape == (n_treated, 1)
    assert result["distances"].shape == (n_treated, 1)

    matched_mask = result["matches"][:, 0] >= 0
    matched_dists = result["distances"][matched_mask]
    assert np.all(matched_dists <= result["caliper_used"])

    # match_rate is the fraction of treated units that found a match
    assert result["match_rate"] == float(matched_mask.mean())
    assert result["n_unmatched"] == int((~matched_mask).sum())

    # estimand string reflects whether any treated units were dropped
    if result["n_unmatched"] > 0:
        assert str(result["estimand"]) == "ATT among matchable units"
    else:
        assert str(result["estimand"]) == "ATT"


def test_causmtchcm_edge():
    """Test edge cases: tighter caliper drops more units than a wider one."""
    rng = np.random.default_rng(42)
    ps = rng.beta(2, 2, 100)
    treat = (rng.random(100) < ps).astype(float)

    wide = causal_caliper_matching(ps, treat, caliper=1.0)
    tight = causal_caliper_matching(ps, treat, caliper=0.001)

    assert tight["n_unmatched"] >= wide["n_unmatched"]
    assert tight["match_rate"] <= wide["match_rate"]
    assert tight["caliper_used"] == 0.001
    assert wide["caliper_used"] == 1.0
