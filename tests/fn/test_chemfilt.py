import math

import pytest

from morie.fn.chemfilt import (
    maccs_fingerprint,
    molecular_properties,
    morgan_environments,
    reos_filter,
    sa_score,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_U32 = 0xFFFFFFFF


def _combine(seed, v):
    return (seed ^ ((v + 0x9E3779B9 + ((seed << 6) & _U32) + (seed >> 2)) & _U32)) & _U32


def test_maccs_matches_the_rdkit_reference_values():
    # the values in the doctest of RDKit's Chem/MACCSkeys.py
    assert maccs_fingerprint("CNO").on_bits == [24, 68, 69, 71, 93, 94, 102, 124, 131, 139, 151, 158, 160, 161, 164]
    assert maccs_fingerprint("CCC").on_bits == [74, 114, 149, 155, 160]


def test_maccs_keys_follow_their_definitions():
    r = maccs_fingerprint(ASPIRIN)
    on = set(r.on_bits)
    assert r.bits[0] == 0 and r.bits[1] == 0  # position 0 unused, key 1 (isotope) undefined
    assert sorted(on) == r.on_bits
    # 6M ring, aromatic, O, C=O, C-O, O>1, O>2, OH
    for key in (163, 162, 164, 154, 157, 159, 146, 139):
        assert key in on
    assert 145 not in on  # only one six-membered ring
    assert 125 not in on  # only one aromatic ring
    assert 166 not in on  # one fragment
    assert 42 not in on and 103 not in on and 88 not in on  # no F, Cl, S
    two = maccs_fingerprint("c1ccc2ccccc2c1")
    assert 125 in two.on_bits and 145 in two.on_bits
    salt = maccs_fingerprint("CC(=O)[O-].[Na+]")
    assert 166 in salt.on_bits and 49 in salt.on_bits and 35 in salt.on_bits


def test_properties_recompute_from_the_published_contributions():
    p = molecular_properties(ASPIRIN)
    weights = {"C": 12.011, "H": 1.008, "O": 15.999}
    mw = 9 * weights["C"] + 4 * weights["O"] + 8 * weights["H"]
    assert pytest.approx(mw, abs=1e-12) == p.MW
    assert p.heavy_atoms == 13 and p.charge == 0
    # Ertl TPSA: ester -O- 9.23, two carbonyl =O 17.07 each, acid -OH 20.23
    assert pytest.approx(9.23 + 2 * 17.07 + 20.23, abs=1e-12) == p.TPSA
    assert (p.HBD, p.HBA, p.Rot) == (1, 3, 2)
    anion = molecular_properties("CC(=O)[O-]")
    assert anion.charge == -1
    assert pytest.approx(17.07 + 23.06, abs=1e-12) == anion.TPSA
    water_like = molecular_properties("OCCO")
    assert water_like.HBD == 2 and water_like.HBA == 2
    assert pytest.approx(2 * 20.23, abs=1e-12) == water_like.TPSA


def test_morgan_identifiers_recompute_from_the_hash():
    # ethane: both carbons share one radius-0 invariant and one radius-1 invariant
    r = morgan_environments("CC", radius=1)
    seed = 0
    for c in (6, 4, 3, 0, 0):  # atomic number, total degree, total H, charge, mass shift (no ring flag)
        seed = _combine(seed, c)
    assert r.counts[seed] == 2
    layer1 = _combine(_combine(0, seed), _combine(_combine(0, 1), seed))
    assert r.counts[layer1] == 1  # the second atom's identical environment is dropped
    assert sorted(r.counts.values()) == [1, 2]
    benzene = morgan_environments("c1ccccc1")
    assert sorted(benzene.counts.values()) == [6, 6, 6]
    assert all(rad <= 2 for _c, _a, rad in benzene.environments)
    assert sum(benzene.counts.values()) == len(benzene.environments)


def test_sa_score_recomputes_from_its_parts():
    fs = {}
    for code in morgan_environments("C1CC2CCC1CC2").counts:
        fs[code] = 0.0
    r = sa_score("C1CC2CCC1CC2", fs)
    n = 8
    assert r.n_bridgehead == 2 and r.n_spiro == 0 and not r.macrocycle
    assert r.fragment_score == 0.0
    assert r.complexity == pytest.approx(-(n**1.005 - n) - math.log10(3), abs=1e-12)
    raw = r.fragment_score + r.complexity + r.symmetry
    assert r.score == pytest.approx(11.0 - (raw + 5.0) / 6.5 * 9.0, abs=1e-12)
    spiro = sa_score("C1CCC2(CC1)CCCC2")
    assert spiro.n_spiro == 1 and spiro.fragment_score == -4.0
    macro = sa_score("C1CCCCCCCCC1")
    assert macro.macrocycle
    chiral = sa_score("N[C@@H](Cc1ccccc1)C(=O)O")
    assert chiral.n_stereo == 1
    hard = sa_score("CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O")
    assert 1.0 <= hard.score <= 10.0 and hard.score > sa_score("CCO").score


def test_reos_filter_alerts_and_property_ranges():
    r = reos_filter(ASPIRIN)
    assert r.passed and r.filter == "OK" and r.alerts == [] and r.violations == []
    halide = reos_filter("CCCCCCCCCCCCBr", rule_sets=("Glaxo",))
    assert not halide.passed
    assert halide.alerts[0] == ("Glaxo", "R1 Reactive alkyl halides")
    assert halide.filter == "R1 Reactive alkyl halides > 0"
    big = reos_filter("CCCCCCCCCCCCCCCCCCCCCCCC", rule_sets=())
    assert set(big.violations) == {"LogP", "Rot"}
    assert reos_filter(ASPIRIN, mw=(0.0, 100.0)).violations == ["MW"]
    both = reos_filter("Oc1ccccc1O", rule_sets=("PAINS", "Dundee"))
    assert both.alerts and all(rs in ("PAINS", "Dundee") for rs, _d in both.alerts)
    assert both.properties["MW"] == pytest.approx(molecular_properties("Oc1ccccc1O").MW)
    with pytest.raises(ValueError):
        reos_filter(ASPIRIN, rule_sets=("Nonesuch",))
