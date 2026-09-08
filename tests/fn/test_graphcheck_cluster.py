"""Graph/identification checker cluster: cmark, exchg, ivcrt, frkst,
chstr, ident, scmdf, ctcfl, potef (+ _dsep helper)."""

from morie.fn import _array_core as np
import pytest

from morie.fn._dsep import d_separated
from morie.fn.chstr import chain_structure
from morie.fn.cmark import causal_markov_condition
from morie.fn.ctcfl import counterfactual_notation
from morie.fn.exchg import exchangeability_assumption
from morie.fn.frkst import fork_structure
from morie.fn.ident import identifiability_conditions
from morie.fn.ivcrt import iv_conditions
from morie.fn.potef import potential_outcomes_individual
from morie.fn.scmdf import scm_definition

CONFOUNDED = {"U": ["T", "Y"], "T": ["Y"]}


def test_dsep_basic():
    chain = {"X": ["Y"], "Y": ["Z"]}
    assert not d_separated(chain, "X", "Z")
    assert d_separated(chain, "X", "Z", ("Y",))
    collider = {"X": ["Z"], "Y": ["Z"]}
    assert d_separated(collider, "X", "Y")
    assert not d_separated(collider, "X", "Y", ("Z",))  # conditioning opens it
    with pytest.raises(ValueError):
        d_separated({"A": ["B"], "B": ["A"]}, "A", "B")  # cycle


def test_cmark_implied_list():
    out = causal_markov_condition({"X": ["Y"], "Y": ["Z"]})
    assert out["implied"] == [("Z", "X", ("Y",))]
    assert out["holds"] is None  # no data supplied


def test_cmark_data_verdicts():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.7, size=n)
    z_ok = y + rng.normal(scale=0.7, size=n)  # true chain
    z_bad = y + 1.0 * x + rng.normal(scale=0.7, size=n)  # extra X -> Z edge
    dag = {"X": ["Y"], "Y": ["Z"]}
    assert causal_markov_condition(dag, {"X": x, "Y": y, "Z": z_ok})["holds"] is True
    bad = causal_markov_condition(dag, {"X": x, "Y": y, "Z": z_bad})
    assert bad["holds"] is False
    assert bad["violations"][0]["pair"] == ("Z", "X")


def test_exchg_backdoor_delegation():
    assert exchangeability_assumption(CONFOUNDED, "T", "Y")["holds"] is False
    assert exchangeability_assumption(CONFOUNDED, "T", "Y", X=("U",))["holds"] is True


def test_ivcrt_valid_and_invalid():
    valid = {"Z": ["X"], "U": ["X", "Y"], "X": ["Y"]}
    out = iv_conditions(valid, "Z", "X", "Y")
    assert out["relevance"] and out["exclusion_independence"] and out["valid"]
    # direct Z -> Y edge breaks exclusion
    direct = {"Z": ["X", "Y"], "U": ["X", "Y"], "X": ["Y"]}
    assert iv_conditions(direct, "Z", "X", "Y")["valid"] is False
    # confounder of Z and Y breaks independence
    conf = {"Z": ["X"], "U": ["X", "Y"], "X": ["Y"], "W": ["Z", "Y"]}
    assert iv_conditions(conf, "Z", "X", "Y")["valid"] is False
    # irrelevant Z fails relevance
    irrel = {"Z": [], "U": ["X", "Y"], "X": ["Y"]}
    out = iv_conditions(irrel | {"Z": ["W2"], "W2": []}, "Z", "X", "Y")
    assert out["relevance"] is False and out["valid"] is False


def test_fork_and_chain_signatures():
    rng = np.random.default_rng(1)
    n = 4000
    b = rng.normal(size=n)
    a = b + rng.normal(scale=0.7, size=n)
    c = b + rng.normal(scale=0.7, size=n)
    f = fork_structure(a, b, c)
    assert f["consistent_with_fork"] is True
    ch = chain_structure(a, b, c)  # Markov-equivalent: same verdict
    assert ch["consistent_with_chain"] is True
    # collider data must fail: A, C independent marginally
    a2 = rng.normal(size=n)
    c2 = rng.normal(size=n)
    b2 = a2 + c2 + rng.normal(scale=0.5, size=n)
    assert fork_structure(a2, b2, c2)["consistent_with_fork"] is False


def test_scmdf_solves_topologically():
    out = scm_definition(
        {"u": 2.0},
        {"X": (("u",), lambda u: u + 1), "Y": (("X",), lambda X: 2 * X)},
    )
    assert out["values"] == {"u": 2.0, "X": 3.0, "Y": 6.0}
    assert out["order"] == ["X", "Y"]
    with pytest.raises(ValueError):
        scm_definition({}, {"A": (("B",), lambda B: B), "B": (("A",), lambda A: A)})


def test_ctcfl_counterfactual():
    eqs = {"X": (("u1",), lambda u1: u1), "Y": (("X", "u2"), lambda X, u2: 2 * X + u2)}
    out = counterfactual_notation({"u1": 1.0, "u2": 0.5}, eqs, "X", 3.0, "Y")
    assert out["factual"] == pytest.approx(2.5)
    assert out["counterfactual"] == pytest.approx(6.5)
    assert out["effect"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        counterfactual_notation({"u1": 1.0}, {"X": (("u1",), lambda u1: u1)}, "u1", 0, "X")


def test_ident_triple():
    T = np.array([1, 0, 1, 0, 1, 1])
    S = np.array(["a", "a", "b", "b", "c", "c"])  # stratum c has no controls
    out = identifiability_conditions(CONFOUNDED, "T", "Y", Z=("U",), treatment=T, strata=S)
    assert out["exchangeability"] is True
    assert out["positivity"] is False
    assert out["empty_arms"] == ["c"]
    assert out["identifiable"] is False
    ok = identifiability_conditions(CONFOUNDED, "T", "Y", Z=("U",), treatment=T[:4], strata=S[:4])
    assert ok["identifiable"] is True
    assert "untestable" in ok["consistency"]


def test_potef_ite_and_selection_bias():
    Y0 = np.array([0.0, 1.0, 2.0, 3.0])
    Y1 = Y0 + 2.0
    out = potential_outcomes_individual(Y1, Y0)
    assert out["ate"] == pytest.approx(2.0)
    assert out["ite"] == pytest.approx([2.0, 2.0, 2.0, 2.0])
    # treated are the high-Y0 units: naive contrast overstates the effect
    T = np.array([0, 0, 1, 1])
    sel = potential_outcomes_individual(Y1, Y0, observed_treatment=T)
    assert sel["naive_diff"] == pytest.approx(4.5 - 0.5)
    assert sel["selection_bias"] == pytest.approx(2.0)
