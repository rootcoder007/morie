"""Tests for the Pearl identification cluster:
causmedb, mdian, backDR, medfm, fdadj, fdcrt, medFront."""

from morie.fn import _array_core as np
import pytest

from morie.fn.backDR import back_door
from morie.fn.causmedb import causal_mediation_baron_kenny
from morie.fn.fdadj import frontdoor_adjustment
from morie.fn.fdcrt import frontdoor_criterion
from morie.fn.mdian import mediation_analysis
from morie.fn.medFront import front_door
from morie.fn.medfm import mediation_formula


def test_mediation_delegates_recover_the_paths():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(size=n)
    y = 0.3 * x + 0.6 * m + rng.normal(size=n)
    r = causal_mediation_baron_kenny(x, m, y)
    assert float(r["a"]) == pytest.approx(0.8, abs=0.15)
    assert float(r["b"]) == pytest.approx(0.6, abs=0.15)
    # mdian with a covariate: residualising a pure-noise covariate must
    # leave the paths essentially unchanged.
    c = rng.normal(size=n)
    r2 = mediation_analysis(y, x, m, X=c)
    assert float(r2["a"]) == pytest.approx(float(r["a"]), abs=0.05)


def test_medfm_matches_hand_computed_cells_and_te_identity():
    """Binary x, binary m, deterministic y = 10x + 5m: every cell mean is
    known, so NDE = 10, NIE = 5 * (P(m=1|x=1) - P(m=1|x=0)), and
    TE = NDE + NIE holds by construction."""
    x = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    y = 10.0 * x + 5.0 * m
    r = mediation_formula(x, m, y, x1=1, x0=0)
    assert float(r["nde"]) == pytest.approx(10.0, abs=1e-12)
    assert float(r["nie"]) == pytest.approx(5.0 * (0.75 - 0.5), abs=1e-12)
    assert float(r["te"]) == pytest.approx(float(r["nde"]) + float(r["nie"]), abs=1e-12)


def test_medfm_linear_dgp_recovers_ab_as_the_indirect_effect():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.integers(0, 2, n)
    m = (0.7 * x + rng.normal(size=n) > 0.3).astype(int)
    y = 2.0 * x + 3.0 * m + rng.normal(size=n)
    r = mediation_formula(x, m, y, x1=1, x0=0)
    p1 = m[x == 1].mean()
    p0 = m[x == 0].mean()
    assert float(r["nde"]) == pytest.approx(2.0, abs=0.2)
    assert float(r["nie"]) == pytest.approx(3.0 * (p1 - p0), abs=0.2)


def test_backdr_delegates_to_the_adjustment_formula():
    z = np.array([0] * 100 + [1] * 100)
    x = np.concatenate([np.repeat([1, 0], [50, 50]), np.repeat([1, 0], [90, 10])])
    y = (z == 1).astype(int)
    r = back_door(y, x, z)
    assert float(r["distribution"]["1"]["1"] if isinstance(r["distribution"], dict) and "1" in r["distribution"] else list(r["distribution"].values())[1][1]) == pytest.approx(0.5, abs=1e-9)


def test_fdadj_cancels_an_unobserved_confounder():
    """Front-door DGP: U confounds X and Y; X -> Z -> Y. The naive
    conditional E[Y=1|X] is confounded, the front-door distribution is
    not. With P(z=1|x)=0.9/0.1 and y depending only on z and u, the
    adjusted P(y=1|do(x=1)) - P(y=1|do(x=0)) approaches the true
    0.8 * 0.4 = 0.32 effect."""
    rng = np.random.default_rng(2)
    n = 40000
    u = rng.integers(0, 2, n)
    x = (rng.random(n) < np.where(u == 1, 0.8, 0.2)).astype(int)
    z = (rng.random(n) < np.where(x == 1, 0.9, 0.1)).astype(int)
    y = (rng.random(n) < 0.1 + 0.4 * z + 0.4 * u).astype(int)
    r = frontdoor_adjustment(x, z, y)
    d = r["distribution"]
    causal = d[1][1] - d[0][1]
    assert causal == pytest.approx(0.8 * 0.4, abs=0.03)
    naive = y[x == 1].mean() - y[x == 0].mean()
    assert abs(naive - 0.32) > abs(causal - 0.32)  # front-door beats naive
    # medFront delegates bit-for-bit.
    r2 = front_door(y, x, z)
    assert r2["distribution"][1][1] == pytest.approx(d[1][1], rel=1e-12)


def test_fdcrt_accepts_the_textbook_graph_and_rejects_violations():
    """Pearl's smoking graph: U -> X, U -> Y, X -> Z -> Y. Z satisfies
    the criterion. Adding a direct X -> Y edge breaks condition 1; a
    U -> Z edge breaks the back-door conditions."""
    g = {"U": ["X", "Y"], "X": ["Z"], "Z": ["Y"]}
    ok = frontdoor_criterion(g, "X", "Y", "Z")
    assert ok["satisfied"] and ok["cond1"] and ok["cond2"] and ok["cond3"]

    direct = {"U": ["X", "Y"], "X": ["Z", "Y"], "Z": ["Y"]}
    r1 = frontdoor_criterion(direct, "X", "Y", "Z")
    assert not r1["satisfied"] and not r1["cond1"]

    confz = {"U": ["X", "Y", "Z"], "X": ["Z"], "Z": ["Y"]}
    r2 = frontdoor_criterion(confz, "X", "Y", "Z")
    assert not r2["satisfied"]

    with pytest.raises(ValueError, match="not a node"):
        frontdoor_criterion(g, "X", "Q", "Z")
