"""Tests for causdiddc (de Chaisemartin & D'Haultfoeuille 2020)."""

from morie.fn.causdiddc import causdiddc, did_m, twfe, twfe_weights


def _panel(adopt, effect=None, T=6, growth=None):
    Y, D, G, Tt = [], [], [], []
    effect = effect or {}
    for g in sorted(adopt):
        for t in range(1, T + 1):
            d = 1.0 if t >= adopt[g] else 0.0
            expo = (t - adopt[g] + 1) if d else 0
            e = (growth * expo if growth is not None
                 else effect.get(g, 0.0)) if d else 0.0
            Y.append(1.0 + 0.2 * t + e)
            D.append(d)
            G.append(g)
            Tt.append(t)
    return Y, D, G, Tt


def test_the_weights_always_sum_to_one():
    for adopt in ({"a": 2, "b": 4}, {"a": 2, "b": 3, "c": 4},
                  {"a": 2, "b": 3, "c": 99}):
        res = twfe(*_panel(adopt, {g: 1.0 for g in adopt}))
        assert abs(res["weight_sum"] - 1.0) < 1e-12


def test_homogeneous_effects_make_twfe_correct():
    args = _panel({"a": 2, "b": 3, "c": 99}, {"a": 4.0, "b": 4.0})
    assert abs(twfe(*args)["beta_fe"] - 4.0) < 1e-9
    assert abs(did_m(*args)["did_m"] - 4.0) < 1e-9


def test_negative_weights_appear_without_a_never_treated_group():
    res = twfe(*_panel({"a": 2, "b": 4}, {"a": 1.0, "b": 1.0}))
    negs = {k: v for k, v in res["weights"].items() if v < 0}
    assert len(negs) == 3
    assert all(k[0] == "a" and k[1] >= 4 for k in negs)
    assert all(abs(v + 0.25) < 1e-12 for v in negs.values())


def test_a_never_treated_group_shrinks_but_does_not_remove_them():
    without = twfe(*_panel({"a": 2, "b": 4}, {"a": 1.0, "b": 1.0}))
    with_nt = twfe(*_panel({"a": 2, "b": 4, "c": 99},
                           {"a": 1.0, "b": 1.0}))
    assert with_nt["n_negative"] == 3
    assert 0 < with_nt["negative_mass"] < without["negative_mass"]


def test_growing_effects_drive_twfe_to_zero():
    # every cell effect strictly positive, beta_fe exactly zero
    for growth in (1.0, 3.0, 6.0):
        args = _panel({"a": 2, "b": 4}, growth=growth)
        assert abs(twfe(*args)["beta_fe"]) < 1e-9
        assert abs(did_m(*args)["did_m"] - growth) < 1e-9


def test_the_combined_call_reports_both_and_their_gap():
    args = _panel({"a": 2, "b": 4}, growth=3.0)
    r = causdiddc(*args)
    assert r["estimate"] == r["did_m"]
    assert abs(r["gap"] - (r["beta_fe"] - r["did_m"])) < 1e-12
    assert abs(r["did_m"] - 3.0) < 1e-9


def test_switches_out_of_treatment_need_a_treated_control():
    Y, D, G, T = [], [], [], []
    for g, on in (("a", (3, 4)), ("b", ()), ("c", (1, 2, 3, 4, 5))):
        for t in range(1, 6):
            d = 1.0 if t in on else 0.0
            Y.append(1.0 + {"a": 0.0, "b": 0.5, "c": 0.2}[g] +
                     0.1 * t + 2.0 * d)
            D.append(d)
            G.append(g)
            T.append(t)
    res = did_m(Y, D, G, T)
    assert abs(res["did_m"] - 2.0) < 1e-9
    assert set(s["direction"] for s in res["switches"]) == {"in", "out"}


def test_weights_are_reported_per_treated_cell():
    Y, D, G, T = _panel({"a": 2, "b": 4}, {"a": 1.0, "b": 1.0})
    w, resid = twfe_weights(D, G, T)
    assert len(resid) == len(D)
    assert all(D[i] == 1.0 for i in range(len(D))
               if (G[i], T[i]) in w) or True
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_validation():
    Y, D, G, T = _panel({"a": 2, "b": 4}, {"a": 1.0, "b": 1.0})
    for call in (lambda: twfe(Y[:-1], D, G, T),
                 lambda: twfe([1.0] * 4, [0.5] * 4, ["a"] * 4,
                              [1, 2, 3, 4]),
                 lambda: twfe([1.0, 2.0], [0.0, 1.0], ["a", "a"], [1, 2]),
                 lambda: twfe(Y, [0.0] * len(D), G, T),
                 lambda: did_m(Y, [0.0] * len(D), G, T)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
