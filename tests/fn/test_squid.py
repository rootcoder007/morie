"""Tests for morie.fn.squid -- threat scoring."""

from morie.fn import _frame_core as pd

from morie.fn._containers import DescriptiveResult
from morie.fn.squid import squid, threat_score


class TestSquid:
    def test_alias(self):
        assert squid is threat_score

    def test_basic(self):
        """Two exactly anti-correlated indicators cancel in the composite."""
        df = pd.DataFrame({"risk1": [1, 2, 3, 4, 5], "risk2": [5, 4, 3, 2, 1]})
        result = threat_score(df)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "Threat Score"
        # z(risk2) = -z(risk1) with equal weights, so every row sums to 0;
        # min == max, so the normaliser returns all zeros.
        assert result.value == 0.0
        assert result.extra["scores"] == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert result.extra["n"] == 5
        assert result.extra["n_features"] == 2
        assert result.extra["normalized"] is True
        # Each column is centred, so its mean z-score -- and hence its
        # contribution -- is zero.
        for col in ("risk1", "risk2"):
            assert abs(result.extra["contributions"][col]) < 1e-12

    def test_single_feature(self):
        """One evenly spaced feature rescales to 0, 1/2, 1."""
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = threat_score(df, features=["x"])
        assert result.extra["n_features"] == 1
        assert result.extra["n"] == 3
        scores = result.extra["scores"]
        assert len(scores) == 3
        for got, want in zip(scores, [0.0, 0.5, 1.0]):
            assert abs(got - want) < 1e-12
        assert abs(result.value - 0.5) < 1e-12

    def test_unnormalized_is_the_raw_weighted_z_sum(self):
        """With normalize=False the scores are the weighted z-scores."""
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
        result = threat_score(df, normalize=False)
        # sample sd of (10, 20, 30) is 10, mean 20, so z = (-1, 0, 1).
        for got, want in zip(result.extra["scores"], [-1.0, 0.0, 1.0]):
            assert abs(got - want) < 1e-12
        assert abs(result.value) < 1e-12
        assert result.extra["normalized"] is False

    def test_weights_scale_the_composite(self):
        """A weight of 3 on the only feature triples the raw composite."""
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
        result = threat_score(df, weights={"x": 3.0}, normalize=False)
        for got, want in zip(result.extra["scores"], [-3.0, 0.0, 3.0]):
            assert abs(got - want) < 1e-12
        # contribution is |mean(z) * w| = 0 for a centred column.
        assert abs(result.extra["contributions"]["x"]) < 1e-12

    def test_missing_feature_is_rejected(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        try:
            threat_score(df, features=["nope"])
        except ValueError as exc:
            assert "nope" in str(exc)
        else:
            raise AssertionError("expected ValueError for a missing feature")
