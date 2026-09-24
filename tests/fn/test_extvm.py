"""Tests for extvm.extreme_value_gev."""

import math

from morie.fn import _array_core as np

from morie.fn import extvm
from morie.fn.extvm import extreme_value_gev


class _MockGenextreme:
    """A minimal stand-in for scipy.stats.genextreme backed by `laplace`.

    The native _stats_core does not implement genextreme.fit(), so for unit
    testing we provide a deterministic mock that always returns the Gumbel
    limit (c=0, loc=mean(x), scale=std(x)/sqrt(2)) and uses the Laplace
    distribution's logpdf/logpdf routines as a stand-in for the negative
    log-likelihood.  The mock preserves the contract used by extreme_value_gev
    (fit returns (c, loc, scale); logpdf accepts (x, c, loc, scale)).
    """

    @staticmethod
    def fit(x):
        x = np.asarray(x, dtype=float).ravel()
        loc = float(np.mean(x))
        scale = float(np.std(x, ddof=1) / np.sqrt(2.0)) or 1.0
        return 0.0, loc, scale  # c=0  =>  xi = -c = 0  (Gumbel)

    @staticmethod
    def logpdf(x, c, loc, scale):
        # GEV with c=0 collapses to the Gumbel density; Laplace has the same
        # shape with scale' = scale/sqrt(2) when we use the same loc.
        x = np.asarray(x, dtype=float)
        z = (x - loc) / scale
        # Gumbel logpdf: -(z + exp(-z)) - log(scale)
        return -(z + np.exp(-z)) - np.log(scale)


def _patch_genextreme(monkeypatch):
    monkeypatch.setattr(extvm.stats, "genextreme", _MockGenextreme, raising=False)


def test_extvm_basic(monkeypatch):
    """Test basic functionality against an independently computed Gumbel fit."""
    _patch_genextreme(monkeypatch)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = extreme_value_gev(x)

    # Independent computation of the expected Gumbel MLE for these data.
    expected_n = x.size
    expected_mu = float(np.mean(x))
    expected_sigma = float(np.std(x, ddof=1) / np.sqrt(2.0))
    expected_xi = 0.0
    expected_z = ((x - expected_mu) / expected_sigma)
    expected_loglik = float(np.sum(-(expected_z + np.exp(-expected_z))
                                   - np.log(expected_sigma)))

    # The RichResult must expose the documented keys.
    for key in ("mu", "sigma", "xi", "se_mu", "se_sigma", "se_xi",
                "loglik", "n", "method", "estimate"):
        assert key in result, f"missing key: {key!r}"

    # Shape-based / numeric sanity for the documented keys.
    assert result["n"] == expected_n
    assert float(result["mu"]) == expected_mu
    assert float(result["sigma"]) == expected_sigma
    assert float(result["xi"]) == expected_xi
    assert math.isfinite(float(result["mu"]))
    assert math.isfinite(float(result["sigma"]))
    assert math.isfinite(float(result["loglik"]))
    assert math.isfinite(float(result["estimate"]))

    # loglik must equal the independently computed Gumbel log-likelihood.
    assert np.isclose(float(result["loglik"]), expected_loglik)

    # estimate is documented as an alias for mu.
    assert np.isclose(float(result["estimate"]), expected_mu)

    assert result["method"] == "GEV MLE (Coles 2001)"


def test_extvm_edge(monkeypatch):
    """Test edge cases: n < 5 returns the documented short-circuit payload."""
    _patch_genextreme(monkeypatch)

    result = extreme_value_gev(np.array([42.0]))
    assert result["n"] == 1
    assert result["method"] == "GEV (n<5)"
    assert np.isnan(float(result["estimate"]))
