"""Tests for causipsw.causal_iptw_attweights."""

from morie.fn import _array_core as np
import pytest

from morie.fn.causipsw import causal_iptw_attweights


def test_causipsw_basic():
    rng = np.random.default_rng(42)
    ps = rng.uniform(0.1, 0.9, 100)
    treat = (rng.random(100) < ps).astype(float)
    result = causal_iptw_attweights(treat, ps)
    w = result["weights"]
    # treated weight exactly 1; control weight e/(1-e)
    assert np.all(w[treat == 1] == 1.0)
    ctrl = treat == 0
    assert w[ctrl] == pytest.approx(ps[ctrl] / (1 - ps[ctrl]))
    assert 0 < result["ess_control"] <= ctrl.sum()


def test_causipsw_edge():
    # hand case: control at e=0.25 gets weight 1/3
    result = causal_iptw_attweights([1, 0], [0.9, 0.25])
    assert result["weights"] == pytest.approx([1.0, 1.0 / 3.0])
    with pytest.raises(ValueError):
        causal_iptw_attweights([0.3, 0.7], [0.5, 0.5])  # non-binary treat
