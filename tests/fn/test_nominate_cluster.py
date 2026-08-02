"""NOMINATE/OC cluster: wnomp, wnoml, ricei, apre, agpar, oclin,
pscrc, brdgo."""

from morie.fn import _array_core as np
import pytest

from morie.fn.agpar import party_unity_score
from morie.fn.apre import oc_apre
from morie.fn.brdgo import bridge_observations
from morie.fn.oclin import oc_cutting_line
from morie.fn.pscrc import pscl_rollcall
from morie.fn.ricei import rice_index
from morie.fn.wnoml import wnominate_logit
from morie.fn.wnomp import wnominate_probability


def test_wnomp_geometry():
    # equidistant -> exactly 0.5; closer to yea -> above 0.5
    mid = wnominate_probability([0.0], [-1.0], [1.0])
    assert mid["p_yea"] == pytest.approx(0.5)
    near = wnominate_probability([-0.9], [-1.0], [1.0])
    assert near["p_yea"] > 0.9
    # beta -> deterministic: probability rises with beta
    lo = wnominate_probability([-0.3], [-1.0], [1.0], beta=1.0)["p_yea"]
    hi = wnominate_probability([-0.3], [-1.0], [1.0], beta=50.0)["p_yea"]
    assert hi > lo
    with pytest.raises(ValueError):
        wnominate_probability([0.0], [-1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        wnominate_probability([0.0], [-1.0], [1.0], beta=0.0)


def test_wnoml_loglik_and_fits():
    # perfectly separated 1-D world: high beta classifies everything
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    Z = np.array([[[-1.0], [1.0]]])  # one roll call, yea at -1, nay at +1
    V = np.array([[1.0], [1.0], [0.0], [0.0]])
    out = wnominate_logit(V, X, Z, beta=50.0)
    assert out["correct_classification"] == pytest.approx(1.0)
    assert out["apre"] == pytest.approx(1.0)
    assert out["gmp"] > 0.9
    assert out["loglik"] < 0  # a log-likelihood, not a probability
    # missing votes are skipped
    V2 = V.copy()
    V2[0, 0] = np.nan
    assert wnominate_logit(V2, X, Z, beta=50.0)["n_choices"] == 3
    with pytest.raises(ValueError):
        wnominate_logit(np.full((2, 1), np.nan), X[:2], Z)


def test_ricei_hand_values():
    # party a: 3-1 on vote 1 -> |0.75 - 0.25| = 0.5; unanimous on vote 2 -> 1
    V = np.array(
        [[1, 1], [1, 1], [1, 1], [0, 1], [1, 0], [0, 1]],
        dtype=float,
    )
    pid = np.array(["a", "a", "a", "a", "b", "b"])
    out = rice_index(V, pid)
    assert out["matrix"]["a"] == pytest.approx([0.5, 1.0])
    assert out["matrix"]["b"] == pytest.approx([0.0, 0.0])  # 1-1 splits
    assert out["by_party"]["a"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        rice_index(V, pid[:3])


def test_apre_footnote_formula():
    # roll call: 6 yea, 4 nay; model errs once -> APRE = 3/4 on that vote
    obs = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=float).T
    pred = obs.copy()
    pred[6, 0] = 1
    out = oc_apre(obs, pred)
    assert out["apre"] == pytest.approx(0.75)
    assert out["minority_total"] == 4
    with pytest.raises(ValueError):
        oc_apre(np.ones((4, 1)), np.ones((4, 1)))  # unanimous


def test_agpar_unity():
    # legislator 3 defects from party a's majority on vote 2
    V = np.array(
        [[1, 1], [1, 1], [1, 0], [0, 0], [0, 0]],
        dtype=float,
    )
    pid = np.array(["a", "a", "a", "b", "b"])
    out = party_unity_score(V, pid)
    assert out["unity"][0] == pytest.approx(1.0)
    assert out["unity"][2] == pytest.approx(0.5)
    assert out["by_party"]["b"] == pytest.approx(1.0)
    # CQ variant: only roll calls where the party majorities oppose
    cq = party_unity_score(V, pid, unity_votes_only=True)
    assert cq["n_votes_scored"][0] == 2  # majorities oppose on both votes
    with pytest.raises(ValueError):
        party_unity_score(V, pid[:2])


def test_oclin_separating_cut():
    # perfectly separable in 1-D: zero errors, cut between 0.1 and 0.4
    x = np.array([-1.0, -0.6, 0.1, 0.4, 0.9])
    v = np.array([1, 1, 1, 0, 0], dtype=float)
    out = oc_cutting_line(x, v)
    assert out["errors"] == 0
    assert 0.1 < out["cutpoint"] < 0.4
    assert out["correct_classification"] == pytest.approx(1.0)
    pred = out["predicted"]
    assert np.nansum(np.abs(pred - v)) == 0
    # 2-D separable case
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X @ np.array([1.0, -0.5]) > 0.2).astype(float)
    out2 = oc_cutting_line(X, y)
    assert out2["correct_classification"] > 0.95
    with pytest.raises(ValueError):
        oc_cutting_line(x, np.ones(5))  # no nays


def test_pscrc_screen():
    raw = np.array(
        [
            [1, 1, 9],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    out = pscl_rollcall(raw, lop=0.1, yea=(1,), nay=(0,), missing=(9,))
    assert np.isnan(out["votes"][0, 2])
    # vote 1: 4-0 unanimous -> dropped; vote 2: 3-1 -> kept at lop=0.1
    assert bool(out["keep"][0]) is False
    assert bool(out["keep"][1]) is True
    assert out["margins"][1] == pytest.approx(0.75)
    assert out["minority"][1] == pytest.approx(0.0)  # nay is the minority side
    assert out["n_dropped"] == 1
    with pytest.raises(ValueError):
        pscl_rollcall(raw, lop=0.7)


def test_brdgo_alignment():
    rng = np.random.default_rng(1)
    ids = [f"L{i}" for i in range(8)]
    A = rng.normal(size=(8, 2))
    th = 0.9
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    B = (A @ R.T) * 2.0 + np.array([3.0, -1.0])  # rotate, scale, shift
    p1 = dict(zip(ids, A))
    p2 = dict(zip(ids, B))
    p2["NEW"] = np.array([0.0, 0.0])
    out = bridge_observations([p1, p2], ids)
    assert out["bridge_residual"] == pytest.approx(0.0, abs=1e-10)
    for b in ids:
        assert out["aligned"][b] == pytest.approx(p1[b], abs=1e-10)
    assert out["scale"] == pytest.approx(0.5, abs=1e-10)  # undoes the x2
    assert "NEW" in out["aligned"]
    with pytest.raises(ValueError):
        bridge_observations([p1, p2], ids[:2])  # too few bridges for 2-D
