"""Anchored tests for phylotr against the printed worked example of
Saitou and Nei (1987), Tables 1-2 and Figure 3, pp. 410-411.

Table 1 distance matrix (8 OTUs). Printed anchors: S0 = 39.28
(275/7); Table 2A cycle-1 S values (S12 = 36.67 minimal); text
p. 411: D1Z = 13, D2Z = 10, L1 = 5, L2 = 2; cycle-2 join {5,6}
with S = 31.30 and L5 = 1, L6 = 4; cycle-3 join of (1-2) with 3
giving L3 = 1 and L(1-2) = 5.5; L7 = 2 and L8 = 6 (Fig. 3f; the
paper notes the S for [1-2-3-4, 5-6] ties with [7, 8], and the
row-major pin selects the pair (7, 8) first).
"""

from morie.fn.phylotr import phylotr, phylogenetic_tree, _sij

D = [
    [0, 7, 8, 11, 13, 16, 13, 17],
    [7, 0, 5, 8, 10, 13, 10, 14],
    [8, 5, 0, 5, 7, 10, 7, 11],
    [11, 8, 5, 0, 8, 11, 8, 12],
    [13, 10, 7, 8, 0, 5, 6, 10],
    [16, 13, 10, 11, 5, 0, 9, 13],
    [13, 10, 7, 8, 6, 9, 0, 8],
    [17, 14, 11, 12, 10, 13, 8, 0],
]


def test_phylotr_saitou_nei_printed_example():
    res = phylotr(D)
    assert abs(res["s0"] - 275.0 / 7.0) < 1e-12  # printed 39.28
    j = res["joins"]
    # cycle 1: OTUs 1 and 2, S12 = 36.67, L1 = 5, L2 = 2
    assert (j[0]["a"], j[0]["b"]) == ("1", "2")
    assert abs(j[0]["S"] - 36.67) < 0.005
    assert abs(j[0]["La"] - 5.0) < 1e-12
    assert abs(j[0]["Lb"] - 2.0) < 1e-12
    # cycle 2: OTUs 5 and 6, printed S = 31.30, L5 = 1, L6 = 4
    assert (j[1]["a"], j[1]["b"]) == ("5", "6")
    assert abs(j[1]["S"] - 31.30) < 0.005
    assert abs(j[1]["La"] - 1.0) < 1e-12
    assert abs(j[1]["Lb"] - 4.0) < 1e-12
    # cycle 3: 3 joins (1-2); printed L3 = 1, L(1-2) = 5.5
    assert {j[2]["a"], j[2]["b"]} == {"3", "(1-2)"}
    lengths3 = {j[2]["a"]: j[2]["La"], j[2]["b"]: j[2]["Lb"]}
    assert abs(lengths3["3"] - 1.0) < 1e-12
    assert abs(lengths3["(1-2)"] - 5.5) < 1e-12
    # cycle 4 joins OTU 4 (L4 = 3, the true-tree value)
    assert "4" in (j[3]["a"], j[3]["b"])
    lengths4 = {j[3]["a"]: j[3]["La"], j[3]["b"]: j[3]["Lb"]}
    assert abs(lengths4["4"] - 3.0) < 1e-9
    # cycle 5: [7, 8], tied with the big-cluster pair (p. 411);
    # printed L7 = 2, L8 = 6
    assert {j[4]["a"], j[4]["b"]} == {"7", "8"}
    lengths5 = {j[4]["a"]: j[4]["La"], j[4]["b"]: j[4]["Lb"]}
    assert abs(lengths5["7"] - 2.0) < 1e-12
    assert abs(lengths5["8"] - 6.0) < 1e-12


def test_phylotr_table2_cycle1_s_values():
    # Table 2A prints all cycle-1 S values to 2 dp; check a spread.
    Dl = [[float(v) for v in row] for row in D]
    printed = {(0, 1): 36.67, (0, 2): 38.33, (1, 2): 38.33,
               (2, 3): 38.67, (4, 5): 37.00, (6, 7): 37.67,
               (3, 4): 39.67, (5, 6): 38.83}
    for (i, jj), s in printed.items():
        assert abs(_sij(Dl, 8, i, jj) - s) < 0.005


def test_phylotr_table2_cycle2_s_values():
    # cycle-2 matrix after joining {1,2} by eq. (5); label order
    # [3, 4, 5, 6, 7, 8, (1-2)]. Table 2B: S(1-2),3 = 31.50,
    # S43 = 32.30, S56 = 31.30, S87 = 31.90, S5,(1-2) = 33.90.
    old = [[float(v) for v in row] for row in D]
    keep = [2, 3, 4, 5, 6, 7]
    D2 = [[old[a][b] for b in keep] + [(old[0][a] + old[1][a]) / 2.0]
          for a in keep]
    D2.append([(old[0][b] + old[1][b]) / 2.0 for b in keep] + [0.0])
    printed = {(0, 6): 31.50, (0, 1): 32.30, (2, 3): 31.30,
               (4, 5): 31.90, (2, 6): 33.90}
    for (i, jj), s in printed.items():
        assert abs(_sij(D2, 7, i, jj) - s) < 0.005


def test_phylotr_additive_four_taxa():
    # additive quartet: tree ((a:1,b:2):1,(c:3,d:4)) hand distances
    Dq = [[0, 3, 5, 6],
          [3, 0, 6, 7],
          [5, 6, 0, 7],
          [6, 7, 7, 0]]
    res = phylotr(Dq, labels=["a", "b", "c", "d"])
    j0 = res["joins"][0]
    assert {j0["a"], j0["b"]} == {"a", "b"}
    assert abs(j0["La"] - 1.0) < 1e-12
    assert abs(j0["Lb"] - 2.0) < 1e-12
    assert phylogenetic_tree is phylotr
