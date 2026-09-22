"""Tests for causipsw0.causal_iptw_atoweights."""

from morie.fn import _array_core as np

from morie.fn.causipsw0 import causal_iptw_atoweights


def test_causipsw0_basic():
    """Test basic ATO (overlap) weights: treated -> 1-e, control -> e.

    Overlap weights are bounded by 1, and the tilting function e(1-e)
    is bounded by 1/4.
    """
    rng = np.random.default_rng(42)
    treat = rng.integers(0, 2, 100).astype(float)
    ps = rng.uniform(0.05, 0.95, 100)

    result = causal_iptw_atoweights(treat, ps, estimand="ato")

    # Documented return keys
    assert isinstance(result, dict)
    for key in ("weights", "estimand", "ess", "max_weight_share", "n_trimmed"):
        assert key in result
    assert result["estimand"] == "ato"
    assert result["n_trimmed"] == 0

    # Documented formula: treated -> 1-e, control -> e
    expected_w = np.where(treat == 1, 1.0 - ps, ps)
    assert np.allclose(result["weights"], expected_w)

    # Overlap weights bounded by 1
    assert result["weights"].max() <= 1.0 + 1e-12

    # Tilting function e(1-e) bounded by 1/4
    assert (ps * (1 - ps)).max() <= 0.25 + 1e-12


def test_causipsw0_edge():
    """Test edge cases: ATE weights (unbounded, dominated), ATT, trimming, stability."""
    rng = np.random.default_rng(42)
    treat = rng.integers(0, 2, 100).astype(float)
    ps = rng.uniform(0.05, 0.95, 100)

    # ATE, unstable: treated -> 1/e, control -> 1/(1-e) — explodes near 0/1
    res_ate = causal_iptw_atoweights(treat, ps, estimand="ate", stabilize=False)
    expected_ate = np.where(treat == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    assert np.allclose(res_ate["weights"], expected_ate)
    # ATE weights are unbounded — must exceed 1 for most realistic ps
    assert res_ate["weights"].max() > 1.0

    # ATT: treated -> 1, control -> e/(1-e)
    res_att = causal_iptw_atoweights(treat, ps, estimand="att", stabilize=False)
    expected_att = np.where(treat == 1, 1.0, ps / (1.0 - ps))
    assert np.allclose(res_att["weights"], expected_att)

    # ATO has strictly smaller max-weight share than ATE (documented property)
    res_ato = causal_iptw_atoweights(treat, ps, estimand="ato")
    assert res_ato["max_weight_share"] < res_ate["max_weight_share"]

    # Trimming drops units and changes the estimand
    res_trim = causal_iptw_atoweights(treat, ps, estimand="ate", trim=0.1)
    expected_dropped = int(((ps < 0.1) | (ps > 0.9)).sum())
    assert res_trim["n_trimmed"] == expected_dropped
    # Trimmed units have zero weight
    mask_dropped = (ps < 0.1) | (ps > 0.9)
    assert np.all(res_trim["weights"][mask_dropped] == 0.0)
