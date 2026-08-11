"""ECFP/FCFP/RDKit fingerprints anchored against RDKit itself.

Every expected number below was produced by RDKit 2025.03 in a scratch
virtualenv (test-only reference use) and is recorded here as a constant, so
the tests carry no rdkit dependency:

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg, rdmolops
    m = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    fp = rfg.GetMorganGenerator(radius=2).GetSparseCountFingerprint(m)
    len(fp.GetNonzeroElements()), sum(fp.GetNonzeroElements().values())

The identifier *values* cannot match RDKit -- our mixer is not
boost::hash_combine, and that departure is documented in each module -- but
the identifier *partition* can and does: the number of distinct identifiers
and the total number of emitted environments agree exactly, at every radius,
on every molecule tested.
"""

from morie.fn.ecfp4 import ecfp4
from morie.fn.ecfp6 import ecfp6
from morie.fn.fcfp4 import fcfp4
from morie.fn.rdkfp import rdkfp


def _mk(n, bonds):
    A = [[0] * n for _ in range(n)]
    for i, j, o in bonds:
        A[i - 1][j - 1] = o
        A[j - 1][i - 1] = o
    return A


# benzene, c1ccccc1
BENZENE = _mk(6, [(1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 6, 4), (6, 1, 4)])
BZ_NUM = [6] * 6
BZ_NH = [1] * 6
BZ_RING = [1] * 6

# ethanol, CCO
ETHANOL = _mk(3, [(1, 2, 1), (2, 3, 1)])
ET_NUM = [6, 6, 8]
ET_NH = [3, 2, 1]

# isobutane, CC(C)C   and cyclopropane, C1CC1
ISOBUTANE = _mk(4, [(1, 2, 1), (2, 3, 1), (2, 4, 1)])
IB_NUM = [6, 6, 6, 6]
IB_NH = [3, 1, 3, 3]
CYCLOPROPANE = _mk(3, [(1, 2, 1), (2, 3, 1), (3, 1, 1)])
CP_NUM = [6, 6, 6]
CP_NH = [2, 2, 2]
CP_RING = [1, 1, 1]

# aspirin, CC(=O)Oc1ccccc1C(=O)O, atoms in RDKit parse order
ASPIRIN = _mk(13, [(1, 2, 1), (2, 3, 2), (2, 4, 1), (4, 5, 1), (5, 6, 4),
                   (6, 7, 4), (7, 8, 4), (8, 9, 4), (9, 10, 4), (10, 5, 4),
                   (10, 11, 1), (11, 12, 2), (11, 13, 1)])
ASP_NUM = [6, 6, 8, 8, 6, 6, 6, 6, 6, 6, 6, 8, 8]
ASP_NH = [3, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
ASP_RING = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
ASP_ARO = ASP_RING

# caffeine, Cn1cnc2c1c(=O)n(C)c(=O)n2C
CAFFEINE = _mk(14, [(1, 2, 1), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 6, 4),
                    (6, 2, 4), (6, 7, 1), (7, 8, 2), (7, 9, 1), (9, 10, 1),
                    (9, 11, 1), (11, 12, 2), (11, 13, 1), (13, 5, 1),
                    (13, 14, 1)])
CAF_NUM = [6, 7, 6, 7, 6, 6, 6, 8, 7, 6, 6, 8, 7, 6]
CAF_NH = [3, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3]
CAF_RING = [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]


def _counts(r):
    return len(r["identifiers"]), r["nenv"]


def test_ecfp4_matches_rdkit_environment_partition():
    # RDKit GetMorganGenerator(radius=2).GetSparseCountFingerprint
    assert _counts(ecfp4(BENZENE, BZ_NUM, numhs=BZ_NH, inring=BZ_RING)) == (3, 18)
    assert _counts(ecfp4(ETHANOL, ET_NUM, numhs=ET_NH)) == (6, 6)
    assert _counts(ecfp4(ISOBUTANE, IB_NUM, numhs=IB_NH)) == (4, 8)
    assert _counts(ecfp4(CYCLOPROPANE, CP_NUM, numhs=CP_NH, inring=CP_RING)) == (3, 7)
    assert _counts(ecfp4(ASPIRIN, ASP_NUM, numhs=ASP_NH, inring=ASP_RING)) == (25, 35)
    assert _counts(ecfp4(CAFFEINE, CAF_NUM, numhs=CAF_NH, inring=CAF_RING)) == (25, 37)


def test_ecfp6_matches_rdkit_environment_partition():
    # RDKit GetMorganGenerator(radius=3).GetSparseCountFingerprint
    assert _counts(ecfp6(BENZENE, BZ_NUM, numhs=BZ_NH, inring=BZ_RING)) == (4, 19)
    assert _counts(ecfp6(ETHANOL, ET_NUM, numhs=ET_NH)) == (6, 6)
    assert _counts(ecfp6(ISOBUTANE, IB_NUM, numhs=IB_NH)) == (4, 8)
    assert _counts(ecfp6(ASPIRIN, ASP_NUM, numhs=ASP_NH, inring=ASP_RING)) == (32, 42)
    assert _counts(ecfp6(CAFFEINE, CAF_NUM, numhs=CAF_NH, inring=CAF_RING)) == (34, 46)


def test_ecfp6_identifiers_contain_ecfp4_identifiers():
    """Radius 3 runs one more round over the same state, so nothing is lost."""
    for adj, num, nh, ring in (
        (ASPIRIN, ASP_NUM, ASP_NH, ASP_RING),
        (CAFFEINE, CAF_NUM, CAF_NH, CAF_RING),
        (BENZENE, BZ_NUM, BZ_NH, BZ_RING),
    ):
        a = set(ecfp4(adj, num, numhs=nh, inring=ring)["identifiers"])
        b = set(ecfp6(adj, num, numhs=nh, inring=ring)["identifiers"])
        assert a <= b


def test_ecfp4_is_invariant_to_atom_renumbering():
    """The defining property of a fingerprint: it does not see the numbering."""
    perm = [12, 0, 5, 3, 9, 2, 7, 1, 11, 4, 10, 8, 6]
    n = len(perm)
    adj = [[ASPIRIN[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
    num = [ASP_NUM[p] for p in perm]
    nh = [ASP_NH[p] for p in perm]
    ring = [ASP_RING[p] for p in perm]
    base = ecfp4(ASPIRIN, ASP_NUM, numhs=ASP_NH, inring=ASP_RING)
    other = ecfp4(adj, num, numhs=nh, inring=ring)
    assert base["identifiers"] == other["identifiers"]
    assert base["bits"] == other["bits"]
    assert sorted(base["count"]) == sorted(other["count"])


def test_ecfp4_benzene_is_one_identifier_per_round():
    """Six equivalent atoms: one identifier value per round, six environments."""
    r = ecfp4(BENZENE, BZ_NUM, numhs=BZ_NH, inring=BZ_RING)
    assert r["nenv"] == 18
    assert len(r["identifiers"]) == 3
    assert sum(r["count"]) == 18
    assert sorted(c for c in r["count"] if c) == [6, 6, 6]


def test_ecfp4_radius_zero_is_the_atom_invariants():
    r = ecfp4(ASPIRIN, ASP_NUM, numhs=ASP_NH, inring=ASP_RING, radius=0)
    assert r["nenv"] == 13
    # RDKit GetMorganGenerator(radius=0): 7 distinct, 13 environments
    assert len(r["identifiers"]) == 7


def test_fcfp4_matches_rdkit_feature_partition():
    """Feature flags as RDKit's GetFeatureInvariants assigns them."""
    # aspirin: acceptors on atoms 3, 4, 12; donor on 13; aromatic 5-10;
    # acidic on 11 (the carboxyl carbon)
    feat = [[0] * 6 for _ in range(13)]
    for i in (3, 4, 12):
        feat[i - 1][1] = 1
    feat[12][0] = 1
    for i in range(5, 11):
        feat[i - 1][2] = 1
    feat[10][5] = 1
    r = fcfp4(ASPIRIN, feat)
    assert _counts(r) == (23, 35)
    # the packed codes are RDKit's GetFeatureInvariants values
    assert r["featurecode"] == [0, 0, 2, 2, 4, 4, 4, 4, 4, 4, 32, 2, 1]

    bz = [[0, 0, 1, 0, 0, 0] for _ in range(6)]
    assert _counts(fcfp4(BENZENE, bz)) == (3, 18)


def test_fcfp4_accepts_packed_codes():
    feat = [[0] * 6 for _ in range(6)]
    for row in feat:
        row[2] = 1
    a = fcfp4(BENZENE, feat)
    b = fcfp4(BENZENE, [4] * 6)
    assert a["identifiers"] == b["identifiers"]


def test_rdkfp_matches_rdkit_subgraph_and_feature_counts():
    """RDKit FindAllSubgraphsOfLengthMToN(m, 1, 7) and GetRDKitFPGenerator."""
    for adj, num, aro, nsub, nfeat in (
        (BENZENE, BZ_NUM, BZ_RING, 31, 6),
        (ETHANOL, ET_NUM, None, 3, 3),
        (ISOBUTANE, IB_NUM, None, 7, 3),
        (CYCLOPROPANE, CP_NUM, None, 7, 3),
        (ASPIRIN, ASP_NUM, ASP_ARO, 301, 201),
    ):
        r = rdkfp(adj, num, aromatic=aro)
        assert r["nsubgraph"] == nsub
        assert r["nfeature"] == nfeat


def test_rdkfp_separates_cyclopropane_from_isobutane():
    """Both have three bonds; the atom count appended to the hash separates
    them, which is exactly why RDKit appends it."""
    c = rdkfp(CYCLOPROPANE, CP_NUM)
    i = rdkfp(ISOBUTANE, IB_NUM)
    assert c["nsubgraph"] == i["nsubgraph"] == 7
    assert set(c["features"]) != set(i["features"])


def test_rdkfp_path_length_window_is_monotone():
    full = rdkfp(ASPIRIN, ASP_NUM, aromatic=ASP_ARO, minpath=1, maxpath=7)
    short = rdkfp(ASPIRIN, ASP_NUM, aromatic=ASP_ARO, minpath=1, maxpath=3)
    assert short["nsubgraph"] < full["nsubgraph"]
    assert set(short["features"]) <= set(full["features"])


def test_rdkfp_linear_paths_are_a_subset_of_subgraphs():
    lin = rdkfp(ASPIRIN, ASP_NUM, aromatic=ASP_ARO, branched=False)
    bra = rdkfp(ASPIRIN, ASP_NUM, aromatic=ASP_ARO, branched=True)
    assert lin["nsubgraph"] <= bra["nsubgraph"]


def test_rdkfp_is_invariant_to_atom_renumbering():
    perm = [12, 0, 5, 3, 9, 2, 7, 1, 11, 4, 10, 8, 6]
    n = len(perm)
    adj = [[ASPIRIN[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
    num = [ASP_NUM[p] for p in perm]
    aro = [ASP_ARO[p] for p in perm]
    a = rdkfp(ASPIRIN, ASP_NUM, aromatic=ASP_ARO)
    b = rdkfp(adj, num, aromatic=aro)
    assert a["features"] == b["features"]
    assert a["bits"] == b["bits"]


def test_shape_and_errors():
    r = ecfp4(ETHANOL, ET_NUM, numhs=ET_NH, nbits=64)
    assert len(r["bits"]) == 64 and len(r["count"]) == 64
    assert r["nset"] == sum(r["bits"])
    for bad in (
        lambda: ecfp4([[0, 1], [1, 0], [0, 0]], [6, 6]),
        lambda: ecfp4(ETHANOL, [6, 6]),
        lambda: ecfp4(ETHANOL, ET_NUM, nbits=0),
        lambda: rdkfp(ETHANOL, ET_NUM, minpath=0),
        lambda: rdkfp(ETHANOL, ET_NUM, minpath=3, maxpath=2),
        lambda: fcfp4(ETHANOL, [[0] * 6, [0] * 6]),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
