"""Tests for crfsel. Full anchor: ledger/wave3/anchor_grf_family.py."""
import pytest
from morie.fn.crfsel import (cate_variable_importance,
                             split_frequency_importance)
from morie.fn.hntfst import grow_forest
from ._grf_fixture import confounded


@pytest.fixture(scope="module")
def d():
    return confounded(700, 21, m_scale=3.0)


def test_the_effect_modifier_is_ranked_first(d):
    """X1 drives the confounding, X2 the effect, X3 nothing."""
    vi = cate_variable_importance(
        d["y"], d["W"], d["X"], n_trees=150, min_leaf=5, seed=2,
        names=["confounder", "modifier", "noise"])
    assert vi["top"] == "modifier"
    assert sum(vi["importance"]) == pytest.approx(1.0, abs=1e-12)
    # Definition 3 puts a floor on every covariate, so a nonzero share
    # is NOT evidence that a covariate is used
    assert vi["importance_by_name"]["noise"] > 0.0


def test_the_depth_decay_changes_the_answer(d):
    trees, _, _ = grow_forest(d["X"], d["y"], n_trees=60, min_leaf=5,
                              seed=3)
    flat = split_frequency_importance(trees, 3, max_depth=4, decay=0.0)
    deep = split_frequency_importance(trees, 3, max_depth=4, decay=4.0)
    assert max(abs(flat[j] - deep[j]) for j in range(3)) > 0.01
    assert sum(flat) == pytest.approx(1.0)
    assert sum(deep) == pytest.approx(1.0)


def test_argument_checks(d):
    with pytest.raises(ValueError):
        cate_variable_importance(d["y"], [1.0] * d["n"], d["X"])
    with pytest.raises(ValueError):
        cate_variable_importance(d["y"], d["W"], d["X"],
                                 names=["only-one"])
    with pytest.raises(ValueError):
        split_frequency_importance([], 3, max_depth=0)
    with pytest.raises(ValueError):
        split_frequency_importance([], 3, decay=-1.0)
