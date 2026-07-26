"""Tests for morie.fn.douml -- Double ML alias (Count Dooku)."""

import pytest

try:
    import doubleml  # noqa: F401

    _HAS_DOUBLEML = True
except ImportError:
    _HAS_DOUBLEML = False

# DoubleML is an optional extra (`pip install morie[doubleml]`), so the two
# tests below that actually fit a model are gated -- a missing optional
# dependency must never be a hard error in the suite. test_is_callable stays
# ungated on purpose: the alias has to exist and be callable whether or not
# the extra is installed, and that is the regression actually worth pinning.
# Matches the pytestmark gate already used in tests/fn/test_dml.py.
requires_doubleml = pytest.mark.skipif(
    not _HAS_DOUBLEML,
    reason="DoubleML not installed (optional extra: pip install morie[doubleml])",
)


class TestDouml:
    def test_is_callable(self):
        """dooku should be a callable alias for estimate_double_ml."""
        from morie.fn.douml import double_ml

        assert callable(double_ml)

    @requires_doubleml
    def test_runs_on_binary_df(self, binary_df):
        """Double ML should run and return a fitted DoubleMLPLR object."""
        from morie.fn.douml import double_ml

        result = double_ml(
            data=binary_df,
            outcome="outcome",
            treatment="treatment",
            covariates=["x1", "x2"],
        )
        # Returns a DoubleMLPLR object
        assert hasattr(result, "coef")
        assert hasattr(result, "se")
        assert hasattr(result, "pval")

    @requires_doubleml
    def test_ate_near_true(self, binary_df):
        """ATE estimate should be in a reasonable range (true effect ~ 0.5)."""
        from morie.fn.douml import double_ml

        result = double_ml(
            data=binary_df,
            outcome="outcome",
            treatment="treatment",
            covariates=["x1", "x2"],
        )
        ate = float(result.coef[0])
        assert -1.0 < ate < 2.0  # reasonable range for true=0.5
