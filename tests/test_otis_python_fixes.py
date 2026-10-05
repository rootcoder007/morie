"""Regressions found while porting the OTIS Ruhela analyses to the R arm."""

from morie import otis_all_analyze as oa
from morie import otis_causal as oc


def test_per_year_driver_asks_for_the_full_battery(monkeypatch):
    seen = {}

    def fake(df, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(oc, "otis_per_year_irm_dml", fake)
    oa.analyze_ruhela_per_year(None, ds_id="x", treatment="t", outcome="y", covariates=["c"])
    # it passed full_ensemble=, which otis_per_year_irm_dml does not take (TypeError every call)
    assert seen["full_battery"] is True
    assert "full_ensemble" not in seen


def test_a01_per_year_uses_the_table_it_is_given(monkeypatch):
    got = {}

    def fake_pair(df=None):
        got["df"] = df
        return "data", "T", "Y", ["c"]

    monkeypatch.setattr(oc, "make_pair_alert_to_volatility_a01", fake_pair)
    monkeypatch.setattr(oa, "analyze_ruhela_per_year", lambda data, **kw: (data, kw["ds_id"]))
    sentinel = object()
    assert oa.analyze_a01_ruhela_per_year(sentinel) == ("data", "a01")
    assert got["df"] is sentinel


def test_interpretation_names_every_estimator():
    import inspect

    src = inspect.getsource(oa.analyze_a01_ruhela_formulations)
    assert "IRM-DML and the MatchIt-then-DoubleML pipeline is the strongest" in src
    assert "(ATE, nonparametric), then-DoubleML" not in src
