"""Anchored tests for phylotr against the printed worked example of
Saitou and Nei (1987), Tables 1-2 and Figure 3, pp. 410-411.

Table 1 distance matrix (8 OTUs). Printed anchors: S0 = 39.28
(275/7), S12 = 36.67, cycle-1 join {1,2} with L1 = 5, L2 = 2
(D1Z = 13, D2Z = 10 stated in text), cycle-2 join {5,6} with
L5 = 1, L6 = 4, cycle-3 join {(1-2),3} with L(1-2) = 5.5, L3 = 1,
final L7 = 2 and L8 = 6.
"""

from morie.fn.phylotr import phylotr, phylogenetic_tree

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
    # cycle 2: OTUs 5 and 6, L5 = 1, L6 = 4 (printed S = 31.30)
    assert (j[1]["a"], j[1]["b"]) == ("5", "6")
    assert abs(j[1]["S"] - 31.30) < 0.005
    assert abs(j[1]["La"] - 1.0) < 1e-12
    assert abs(j[1]["Lb"] - 4.0) < 1e-12
    # cycle 3: (1-2) and 3, L(1-2) = 5.5, L3 = 1 (printed S = 31.50)
    assert (j[2]["a"], j[2]["b"]) == ("(1-2)", "3")
    assert abs(j[2]["S"] - 31.50) < 0.005
    assert abs(j[2]["La"] - 5.5) < 1e-12
    assert abs(j[2]["Lb"] - 1.0) < 1e-12
    # cycle 4: joins OTU 4; cycle 5 joins the two big clusters
    assert j[3]["b"] == "4"
    # final three branches contain L7 = 2 and L8 = 6 (Fig. 3f)
    fin = dict(zip(res["final_labels"], res["final_lengths"]))
    assert abs(fin["7"] - 2.0) < 1e-12
    assert abs(fin["8"] - 6.0) < 1e-12


def test_phylotr_table2_s_values():
    # Table 2A prints all cycle-1 S values to 2 dp; check a spread.
    res = phylotr(D)
    from morie.fn.phylotr import _sij
    Dl = [[float(v) for v in row] for row in D]
    printed = {(0, 1): 36.67, (0, 2): 38.33, (1, 2): 38.33,
               (2, 3): 38.67, (4, 5): 37.00, (6, 7): 37.67,
               (3, 4): 39.67, (5, 6): 38.83}
    for (i, jj), s in printed.items():
        assert abs(_sij(Dl, 8, i, jj) - s) < 0.005
    assert res["n"] == 8


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
