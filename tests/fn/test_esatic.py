"""Tests for esatic.eap_information."""

from morie.fn import _array_core as np

from morie.fn.esatic import eap_information


def test_esatic_basic():
    """Test basic functionality with a single 2PL item."""
    # One item: (a, b, c, d) = (1.0, 0.0, 0.0, 1.0), a 2PL with
    # discrimination 1 and difficulty 0.  Response is 1 (correct).
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    x = np.array([1.0])
    result = eap_information(items, x, D=1.0, prior_mean=0.0, prior_sd=1.0,
                            lower=-4.0, upper=4.0, nqp=33)
    # The function returns a RichResult, not a dict.
    # Verify the documented return keys are present.
    for key in ("estimate", "se", "information", "se_ml",
                "item_information", "prob", "J", "nqp"):
        assert key in result, f"missing key: {key}"

    # Documented shapes/values.
    assert result["J"] == 1.0
    assert result["nqp"] == 33.0
    assert len(result["item_information"]) == 1
    assert len(result["prob"]) == 1

    # With prior N(0, 1), a single easy item answered correctly, the EAP
    # should be positive but bounded inside the prior range.
    assert 0.0 < result["estimate"] < 1.0
    # Posterior sd is strictly positive and below the prior sd.
    assert 0.0 < result["se"] < 1.0
    # se_ml = 1/sqrt(information); both must agree on finiteness.
    assert result["se_ml"] > 0.0
    assert result["information"] > 0.0


def test_esatic_edge():
    """Test edge cases: responses all incorrect, still well-defined."""
    # Three 2PL items at various difficulties, all answered incorrectly.
    # The EAP should be negative (likelihood peaks below zero) and finite.
    items = np.array([
        [1.0, -1.0, 0.0, 1.0],
        [1.0,  0.0, 0.0, 1.0],
        [1.0,  1.0, 0.0, 1.0],
    ])
    x = np.array([0.0, 0.0, 0.0])
    result = eap_information(items, x)

    for key in ("estimate", "se", "information", "se_ml",
                "item_information", "prob", "J", "nqp"):
        assert key in result, f"missing key: {key}"

    assert result["J"] == 3.0
    assert len(result["item_information"]) == 3
    # All wrong with a 2PL symmetric around b=0 -> EAP below 0.
    assert result["estimate"] < 0.0
    # Information at the EAP is positive and finite; se_ml is finite.
    assert result["information"] > 0.0
    assert result["se_ml"] > 0.0
