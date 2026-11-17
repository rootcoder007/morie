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


def test_ckan_lookup_uses_the_package_id(monkeypatch):
    # the dataset was renamed; its old address answers 404, its id does not change
    import io
    import json
    import urllib.request

    from morie import otis_datasets as od

    seen = []
    body = {"success": True, "result": {"resources": [{"name": "a01_x", "url": "https://h/a01_x.csv"}]}}

    def fake_open(url, timeout=None):
        seen.append(url)
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    assert od._ckan_resource_index() == {"a01_x": "https://h/a01_x.csv"}
    assert seen == [od.OTIS_CKAN_BASE + "/package_show?id=09f7fc65-d3bb-4ca8-8b84-1cdc3ef73c36"]
