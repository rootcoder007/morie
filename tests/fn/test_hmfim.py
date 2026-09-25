"""Tests for hmfim.geron_feature_importance."""

import pytest

from morie.fn.hmfim import geron_feature_importance


RAW = [[0.30896362200710026, 0.04243386243386245, 0.0382321451886669],
       [0.29358730158730156, 0.09100529100529102, 0.06556613756613756],
       [0.2968574635241302, 0.0927877677877678, 0.004585537918871241]]


def test_hmfim_basic():
    """Per-tree raw decreases (sklearn tree_.compute_feature_importances(
    normalize=False) of a 3-tree RandomForestClassifier) give the forest's
    feature_importances_."""
    r = geron_feature_importance(RAW)
    sk = [0.7327193154552024, 0.18214498879753585, 0.08513569574726172]
    for got, want in zip(r["importance"].tolist(), sk):
        assert got == pytest.approx(want, rel=1e-12)
    assert r["ranking"] == ["x0", "x1", "x2"]
    s = r["importance"].tolist()
    assert r["effective_features"] == pytest.approx(1 / sum(v * v for v in s), rel=1e-12)


def test_hmfim_edge():
    """Equal shares have Gini 0 and p effective features; raw means are
    returned unscaled when normalize=False."""
    r = geron_feature_importance([[0.2, 0.2, 0.2, 0.2]])
    assert r["concentration"] == pytest.approx(0.0, abs=1e-15)
    assert r["effective_features"] == pytest.approx(4.0, rel=1e-15)
    raw = geron_feature_importance(RAW, normalize=False)["importance"].tolist()
    for j in range(3):
        assert raw[j] == pytest.approx(sum(t[j] for t in RAW) / 3, rel=1e-15)
    with pytest.raises(ValueError, match="negative"):
        geron_feature_importance([[0.5, -0.1]])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmfim as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
