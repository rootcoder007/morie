"""Tests for morie.fn.fzinf -- Mamdani fuzzy inference (Mamdani & Assilian 1975)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.fzinf import fzinf


def _rule(ant, cons, **kw):
    return {"antecedent": ant, "consequent": cons, **kw}


def test_fzinf_single_rule_centroid_is_the_consequent_centroid():
    """One rule firing fully: the aggregate IS the consequent, so the centroid
    is the consequent's own centroid. A symmetric triangle on (0, 0.5, 1)
    centres at 0.5."""
    r = fzinf(
        rules=[_rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.0, 0.5, 1.0)))],
        inputs=[0.5],
    )
    assert r.value == pytest.approx(0.5, abs=1e-3)


def test_fzinf_asymmetric_consequent_shifts_the_output():
    """A triangle centred at 0.8 must defuzzify above 0.5, not at it."""
    r = fzinf(
        rules=[_rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.6, 0.8, 1.0)))],
        inputs=[0.5],
    )
    assert r.value == pytest.approx(0.8, abs=1e-2)


def test_fzinf_firing_strength_is_the_min_over_antecedents():
    """Mamdani conjunction is min. With antecedent memberships 1.0 and 0.5,
    the rule fires at 0.5, so the consequent is clipped at 0.5."""
    r = fzinf(
        rules=[
            _rule(
                [
                    (0, "triangular", (0.0, 0.5, 1.0)),   # at 0.50 -> 1.0
                    (1, "triangular", (0.0, 0.5, 1.0)),   # at 0.75 -> 0.5
                ],
                ("triangular", (0.0, 0.5, 1.0)),
            )
        ],
        inputs=[0.5, 0.75],
    )
    assert float(np.max(r.extra["aggregate_mf"])) == pytest.approx(0.5, abs=1e-2)


def test_fzinf_weight_scales_the_firing_strength():
    full = fzinf(
        rules=[_rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.0, 0.5, 1.0)))],
        inputs=[0.5],
    )
    half = fzinf(
        rules=[
            _rule(
                [(0, "triangular", (0.0, 0.5, 1.0))],
                ("triangular", (0.0, 0.5, 1.0)),
                weight=0.5,
            )
        ],
        inputs=[0.5],
    )
    assert float(np.max(half.extra["aggregate_mf"])) == pytest.approx(
        0.5 * float(np.max(full.extra["aggregate_mf"])), abs=1e-2
    )


def test_fzinf_aggregation_across_rules_is_max():
    """Two rules with disjoint consequents both survive aggregation, pulling
    the centroid between them."""
    r = fzinf(
        rules=[
            _rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.0, 0.1, 0.2))),
            _rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.8, 0.9, 1.0))),
        ],
        inputs=[0.5],
    )
    assert r.value == pytest.approx(0.5, abs=0.05)
    assert r.extra["n_rules"] == 2


def test_fzinf_mom_differs_from_centroid_on_a_skewed_aggregate():
    """Mean-of-maxima reports where the membership peaks; the centroid reports
    the balance point. On a two-humped aggregate they must differ."""
    rules = [
        _rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.0, 0.05, 0.1))),
        _rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.2, 0.9, 1.0))),
    ]
    centroid = fzinf(rules=rules, inputs=[0.5], defuzz_method="centroid").value
    mom = fzinf(rules=rules, inputs=[0.5], defuzz_method="mom").value
    assert centroid != pytest.approx(mom, abs=1e-3)


def test_fzinf_no_rule_fires_falls_back_to_the_universe_mean():
    """Zero total area cannot be divided by; the documented fallback is the
    midpoint of the output universe."""
    r = fzinf(
        rules=[_rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.0, 0.5, 1.0)))],
        inputs=[5.0],  # far outside the antecedent support -> fires at 0
        universe=(0.0, 1.0),
    )
    assert r.extra["total_area"] == pytest.approx(0.0, abs=1e-12)
    assert r.value == pytest.approx(0.5, abs=1e-6)


def test_fzinf_respects_a_custom_universe():
    r = fzinf(
        rules=[_rule([(0, "triangular", (0.0, 0.5, 1.0))], ("triangular", (0.0, 5.0, 10.0)))],
        inputs=[0.5],
        universe=(0.0, 10.0),
    )
    assert r.value == pytest.approx(5.0, abs=1e-2)
    assert r.extra["output_universe"][-1] == pytest.approx(10.0)


def test_fzinf_rejects_a_tuple_rule_with_a_message_that_names_the_shape():
    """rules is a list of DICTS. Passing tuples used to die on
    `'tuple' object has no attribute 'get'` deep in the inference loop,
    naming neither the argument nor the shape it wanted -- and the test that
    should have caught it was marked xfail for an unrelated reason.
    """
    with pytest.raises(TypeError, match="must be a dict"):
        fzinf(rules=[(0.7, 0.3)], inputs=[0.5, 0.5])


def test_fzinf_rejects_a_rule_missing_required_keys():
    with pytest.raises(ValueError, match="missing required key"):
        fzinf(rules=[{"antecedent": [(0, "triangular", (0.0, 0.5, 1.0))]}], inputs=[0.5])


def test_fzinf_rejects_empty_rules():
    with pytest.raises(ValueError, match="must not be empty"):
        fzinf(rules=[], inputs=[0.5])


def test_cheatsheet():
    from morie.fn.fzinf import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
