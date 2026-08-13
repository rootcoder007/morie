"""Tests for gwPot.

The generated tests these replace passed a 100-long normal draw as the
gas name and as the horizon, so both raised before reaching anything.
Every assertion here is against a value printed in IPCC AR6 WG1 Chapter
7 Supplementary Material Table 7.SM.7, or against a property of the
table that a transcription slip would break.
"""

import pytest

from morie.fn.gwPot import gwPot, global_warming_potential


def test_printed_gwp_values():
    """The assessed GWPs, exactly as Table 7.SM.7 prints them."""
    assert gwPot("CH4", 100)["estimate"] == 27.9
    assert gwPot("CH4", 20)["estimate"] == 81.2
    assert gwPot("N2O", 100)["estimate"] == 273.0
    assert gwPot("SF6", 100)["estimate"] == 24300.0
    assert gwPot("CFC-12", 20)["estimate"] == 12700.0


def test_co2_is_the_unit_at_every_horizon():
    for h in (20, 100, 500):
        assert gwPot("CO2", h)["estimate"] == 1.0
        r = gwPot("CO2", h)
        assert r["gwp_from_agwp"] == 1.0


def test_gwp_reproduces_agwp_ratio():
    """GWP_H = AGWP_x(H) / AGWP_CO2(H) -- the definition, Sec. 7.SM.5.

    This is the check that catches a mistyped AGWP: the printed GWP and
    the ratio of the printed AGWPs are independent entries in the table,
    so they only agree if both were transcribed correctly. AR6 prints
    three significant figures, so 1% is the tightest tolerance the
    rounding permits.
    """
    for gas in ("CH4", "N2O", "CFC-11", "CFC-12", "HFC-134a", "SF6"):
        for h in (20, 100, 500):
            r = gwPot(gas, h)
            assert r["gwp_from_agwp"] == pytest.approx(r["estimate"],
                                                       rel=0.01), (gas, h)


def test_lifetime_orders_the_horizon_response():
    """Short-lived gases lose GWP with horizon, long-lived ones gain it.

    CH4 (11.8 yr) decays well inside 100 years, so its pulse forcing
    stops accumulating while CO2's keeps going; SF6 (1000 yr) does not.
    A table with the columns swapped would fail this.
    """
    ch4 = [gwPot("CH4", h)["estimate"] for h in (20, 100, 500)]
    sf6 = [gwPot("SF6", h)["estimate"] for h in (20, 100, 500)]
    assert ch4[0] > ch4[1] > ch4[2]
    assert sf6[0] < sf6[1] < sf6[2]
    assert gwPot("CH4", 100)["lifetime"] == 11.8
    assert gwPot("SF6", 100)["lifetime"] == 1000.0


def test_case_insensitive_and_alias():
    assert gwPot("ch4", 100)["estimate"] == gwPot("CH4", 100)["estimate"]
    assert global_warming_potential is gwPot


def test_rejects_unknown_gas_and_horizon():
    with pytest.raises(ValueError):
        gwPot("argon", 100)
    with pytest.raises(ValueError):
        gwPot("CH4", 50)
