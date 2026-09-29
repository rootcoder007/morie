import pytest

from morie.fn.chemsmarts import _CRIPPEN, _PAINS, _molecule, clogp_estimate, pains_filter, smarts_match

TAB = {}
for name, sm, lp, mr in _CRIPPEN:
    TAB.setdefault(name, (lp, mr))


def test_crippen_ethanol_by_hand():
    # CH3 (C1), CH2 bonded to O (C3), OH (O2), five H on carbon (H1), H on alcohol O next to CX4 (H2)
    types = ["C1", "C3", "O2"] + ["H1"] * 5 + ["H2"]
    r = clogp_estimate("CCO")
    assert sorted(r.types) == sorted(types)
    assert r.logp == pytest.approx(sum(TAB[t][0] for t in types), abs=1e-12)
    assert r.mr == pytest.approx(sum(TAB[t][1] for t in types), abs=1e-12)


def test_crippen_phenol_and_kekule_input_agree():
    r = clogp_estimate("c1ccccc1O")
    types = ["C18"] * 5 + ["C23", "O2"] + ["H1"] * 5 + ["H2"]
    assert sorted(r.types) == sorted(types)
    assert r.logp == pytest.approx(sum(TAB[t][0] for t in types), abs=1e-12)
    k = clogp_estimate("C1=CC=CC=C1O")
    assert k.logp == pytest.approx(r.logp, abs=1e-12) and k.types == r.types


def test_aromaticity_perception():
    m = _molecule("O=C1NC(=O)C=CN1")  # uracil: aromatic (RDKit model)
    assert all(m["arom"][i] for i in (1, 2, 3, 5, 6, 7))
    q = _molecule("O=C1C=CC(=O)C=C1")  # benzoquinone: not aromatic
    assert not any(q["arom"])
    t = _molecule("s1(c2c(c(c1)C)cccc2)(=O)=O")  # benzothiophene dioxide: only the benzene ring
    assert sum(t["arom"]) == 6 and not t["arom"][0]
    assert _molecule("c1ccccc1")["h"] == [1] * 6


def test_smarts_matching():
    assert smarts_match("CC(=O)Oc1ccccc1C(=O)O", "[CX3](=O)[OX2H1]").anchors == [10]
    assert smarts_match("CC(=O)Nc1ccccc1", "[#6;R]!@[#7;!H0]").anchors == [4]
    assert smarts_match("CCO", "[OH]").anchors == [2]
    assert smarts_match("CCN(C)C", "[$([NX3](C)(C)C)]").anchors == [2]
    assert not smarts_match("c1ccccc1", "C").matched
    assert smarts_match("c1ccccc1CC(=O)O", "c.C(=O)[OH]").matched
    assert smarts_match("CCO[H]", "[#1]", merge_hs=False).anchors == [3]
    with pytest.raises(ValueError):
        smarts_match("CCO", "[C")


@pytest.mark.parametrize(
    "smiles,idx",
    [
        ("S=C(N/N=C/c1ccc2ccccc2c1)Nc1ccccc1", 61),
        ("Cc1ccc(Nc2ccc(N)cn2)cc1", 249),
        ("CCCCCCC(C)Nc1ccc(Nc2ccccc2)cc1", 124),
        ("Oc1ccc(CC)cc1O", None),
    ],
)
def test_pains_known_hits(smiles, idx):
    r = pains_filter(smiles)
    if idx is None:
        assert r.names == ["catechol_A(92)"]
    else:
        assert _PAINS[idx][0] in r.names
    assert r.n_alerts == len(r.names) and r.flagged


def test_pains_clean_molecule():
    assert not pains_filter("CC(=O)Oc1ccccc1C(=O)O").flagged
