from morie.fn.moldesc import (
    lipinski_descriptors,
    smiles_hba,
    smiles_hbd,
    smiles_molecular_weight,
    smiles_rotatable_bonds,
    smiles_tpsa,
)

M = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.067, "Cl": 35.453}


def test_molecular_weight_from_formula():
    assert abs(smiles_molecular_weight("CC(=O)Oc1ccccc1C(=O)O") - (9 * M["C"] + 8 * M["H"] + 4 * M["O"])) < 1e-9
    assert (
        abs(
            smiles_molecular_weight("Cn1cnc2c1c(=O)n(C)c(=O)n2C") - (8 * M["C"] + 10 * M["H"] + 4 * M["N"] + 2 * M["O"])
        )
        < 1e-9
    )
    assert (
        abs(
            smiles_molecular_weight("O=S(=O)(N)c1ccc(Cl)cc1")
            - (6 * M["C"] + 6 * M["H"] + M["N"] + 2 * M["O"] + M["S"] + M["Cl"])
        )
        < 1e-9
    )


def test_tpsa_by_fragment_table():
    # aspirin: two carbonyl O (17.07), one ester O (9.23), one hydroxyl O (20.23)
    assert abs(smiles_tpsa("CC(=O)Oc1ccccc1C(=O)O").tpsa - (2 * 17.07 + 9.23 + 20.23)) < 1e-12
    # caffeine: three aromatic N with a methyl (4.93), one aromatic N (12.89), two exocyclic C=O (17.07)
    assert abs(smiles_tpsa("Cn1cnc2c1c(=O)n(C)c(=O)n2C").tpsa - (3 * 4.93 + 12.89 + 2 * 17.07)) < 1e-12
    # zwitterionic glycine: NH3+ (27.64), carbonyl O (17.07), O- (23.06)
    assert abs(smiles_tpsa("[NH3+]CC([O-])=O").tpsa - (27.64 + 17.07 + 23.06)) < 1e-12
    assert abs(smiles_tpsa("C1CO1").tpsa - 12.53) < 1e-12  # three-ring ether


def test_counts():
    assert smiles_hba("CC(=O)NC1=CC=C(O)C=C1") == 3 and smiles_hbd("CC(=O)NC1=CC=C(O)C=C1") == 2
    assert smiles_hbd("OCC(O)CO") == 3
    assert smiles_rotatable_bonds("CC(=O)NC1=CC=C(O)C=C1") == 1  # amide C-N excluded
    assert smiles_rotatable_bonds("CC(=O)NC1=CC=C(O)C=C1", exclude_amide=False) == 2
    assert smiles_rotatable_bonds("CC(C)Cc1ccc(C(C)C(=O)O)cc1") == 4
    r = lipinski_descriptors("CCOC(=O)c1ccc(N)cc1")
    assert (r.hba, r.hbd, r.rotatable_bonds) == (3, 2, 3)
