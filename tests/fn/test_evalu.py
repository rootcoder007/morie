"""Tests for the E-value cluster (evalu, evaltw, causfromle, ucbias)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.causfromle import causal_e_value
from morie.fn.evaltw import e_value_unmeasured_confounding
from morie.fn.evalu import evalue
from morie.fn.ucbias import unmeasured_conf_bias


def test_evalue_matches_the_papers_worked_values():
    """RR = 2 gives E = 2 + sqrt(2) = 3.41; RR = 1 needs no confounding
    at all (VanderWeele & Ding 2017)."""
    assert float(evalue(2.0)["evalue"]) == pytest.approx(2 + np.sqrt(2), rel=1e-12)
    assert float(evalue(1.0)["evalue"]) == pytest.approx(1.0, abs=1e-12)


def test_evalue_is_symmetric_in_protective_and_harmful_directions():
    """RR and 1/RR describe the same strength of association, so their
    E-values coincide."""
    a = float(evalue(2.5)["evalue"])
    b = float(evalue(1 / 2.5)["evalue"])
    assert a == pytest.approx(b, rel=1e-12)


def test_evalue_ci_uses_the_limit_closer_to_the_null():
    r = evalue(3.0, ci_lower=1.5, ci_upper=6.0)
    assert float(r["evalue_ci"]) == pytest.approx(1.5 + np.sqrt(1.5 * 0.5), rel=1e-12)
    # A CI crossing 1 needs no confounding to explain away significance.
    assert float(evalue(1.8, ci_lower=0.9, ci_upper=3.6)["evalue_ci"]) == 1.0


def test_evalue_front_ends_delegate_bit_for_bit():
    a = evalue(2.2, ci_lower=1.3, ci_upper=3.7)
    b = e_value_unmeasured_confounding(2.2, 1.3, 3.7)
    assert float(a["evalue"]) == float(b["evalue"])
    assert float(a["evalue_ci"]) == float(b["evalue_ci"])
    assert float(causal_e_value(2.2)["evalue"]) == float(evalue(2.2)["evalue"])


def test_ucbias_bounding_factor_and_the_evalue_connection():
    """B = RR_UD RR_UY / (RR_UD + RR_UY - 1): equal associations of 2
    give B = 4/3. And on the diagonal, plugging the E-value of an
    observed RR back in reproduces that RR -- the defining link between
    the two quantities (Ding & VanderWeele 2016)."""
    assert float(unmeasured_conf_bias(2.0, 2.0)["bias_factor"]) == pytest.approx(4 / 3, rel=1e-12)
    rr = 2.5
    e = float(evalue(rr)["evalue"])
    B = float(unmeasured_conf_bias(e, e)["bias_factor"])
    assert B == pytest.approx(rr, rel=1e-9)
    r = unmeasured_conf_bias(e, e, RR_obs=rr)
    assert r["explains_away"] is True
    weak = unmeasured_conf_bias(1.3, 1.3, RR_obs=rr)
    assert weak["explains_away"] is False
    assert float(weak["rr_bound"]) > 1.0


def test_evalue_cluster_validates_input():
    with pytest.raises(ValueError, match="positive"):
        evalue(-1.0)
    with pytest.raises(ValueError, match="both"):
        evalue(2.0, ci_lower=1.5)
    with pytest.raises(ValueError, match=">= 1"):
        unmeasured_conf_bias(0.8, 2.0)
