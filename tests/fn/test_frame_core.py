"""Equivalence tests: morie.fn._frame_core vs pandas."""

import math

import pytest

real_pd = __import__("pytest").importorskip("pandas")

from morie.fn import _frame_core as pd


DATA = {
    "g": ["a", "b", "a", "b", "a", "c"],
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "y": [2.0, float("nan"), 6.0, 8.0, 10.0, 12.0],
}


def _pair():
    return pd.DataFrame({k: list(v) for k, v in DATA.items()}), \
        real_pd.DataFrame({k: list(v) for k, v in DATA.items()})


def test_series_reductions_match():
    s = pd.Series([1.0, 2.0, float("nan"), 4.0, 10.0])
    w = real_pd.Series([1.0, 2.0, float("nan"), 4.0, 10.0])
    assert s.sum() == w.sum()
    assert s.mean() == w.mean()
    assert s.std() == pytest.approx(w.std(), rel=1e-12)
    assert s.var() == pytest.approx(w.var(), rel=1e-12)
    assert s.median() == w.median()
    assert s.quantile(0.25) == pytest.approx(w.quantile(0.25))
    assert s.count() == w.count()
    assert s.min() == w.min() and s.max() == w.max()


def test_series_ops_and_masking():
    s = pd.Series([1.0, 5.0, 3.0])
    w = real_pd.Series([1.0, 5.0, 3.0])
    assert (s * 2 + 1).tolist() == (w * 2 + 1).tolist()
    assert s[(s > 2).tolist()].tolist() == w[w > 2].tolist()
    assert s.clip(2.0, 4.0).tolist() == w.clip(2.0, 4.0).tolist()
    assert s.rank().tolist() == w.rank().tolist()
    assert s.diff().tolist()[1:] == w.diff().tolist()[1:]
    assert s.cumsum().tolist() == w.cumsum().tolist()


def test_dataframe_basics_match():
    g, w = _pair()
    assert g.shape == w.shape
    assert list(g.columns) == list(w.columns)
    assert g["x"].mean() == w["x"].mean()
    assert g.dropna().shape == w.dropna().shape
    assert g.fillna(0.0)["y"].sum() == w.fillna(0.0)["y"].sum()
    gm = g[(g["x"] > 2.5).tolist()]
    wm = w[w["x"] > 2.5]
    assert gm.shape == wm.shape
    assert gm["y"].tolist()[0] == wm["y"].tolist()[0]


def test_sort_values_and_iloc():
    g, w = _pair()
    gs = g.sort_values("x", ascending=False)
    ws = w.sort_values("x", ascending=False)
    assert gs["x"].tolist() == ws["x"].tolist()
    assert g.iloc[2]["x"] == w.iloc[2]["x"]
    assert g.iloc[slice(1, 4)]["x"].tolist() == \
        w.iloc[1:4]["x"].tolist()


def test_groupby_matches():
    g, w = _pair()
    gm = g.groupby("g")["x"].mean()
    wm = w.groupby("g")["x"].mean()
    assert gm.tolist() == wm.tolist()
    assert list(gm.index) == list(wm.index)
    gs = g.groupby("g").sum()
    ws = w.groupby("g")[["x", "y"]].sum()
    assert gs["x"].tolist() == ws["x"].tolist()
    assert gs["y"].tolist() == ws["y"].tolist()
    ga = g.groupby("g").agg({"x": "mean", "y": "max"})
    wa = w.groupby("g").agg({"x": "mean", "y": "max"})
    assert ga["x"].tolist() == wa["x"].tolist()
    assert ga["y"].tolist() == wa["y"].tolist()


def test_merge_matches():
    left_d = {"k": ["a", "b", "c"], "v": [1, 2, 3]}
    right_d = {"k": ["a", "b", "d"], "u": [10, 20, 40]}
    gl = pd.DataFrame(left_d).merge(pd.DataFrame(right_d), on="k")
    wl = real_pd.merge(real_pd.DataFrame(left_d),
                       real_pd.DataFrame(right_d), on="k")
    assert gl.shape == wl.shape
    assert gl["v"].tolist() == wl["v"].tolist()
    assert gl["u"].tolist() == wl["u"].tolist()
    go = pd.DataFrame(left_d).merge(pd.DataFrame(right_d), on="k",
                                    how="left")
    wo = real_pd.merge(real_pd.DataFrame(left_d),
                       real_pd.DataFrame(right_d), on="k", how="left")
    assert go.shape == wo.shape
    assert math.isnan(go["u"].tolist()[2]) \
        and math.isnan(wo["u"].tolist()[2])


def test_concat_matches():
    a = {"x": [1, 2], "y": [3, 4]}
    b = {"x": [5], "y": [6]}
    gc = pd.concat([pd.DataFrame(a), pd.DataFrame(b)],
                   ignore_index=True)
    wc = real_pd.concat([real_pd.DataFrame(a), real_pd.DataFrame(b)],
                        ignore_index=True)
    assert gc["x"].tolist() == wc["x"].tolist()
    assert gc["y"].tolist() == wc["y"].tolist()


def test_crosstab_matches():
    g, w = _pair()
    h = ["u", "v", "u", "v", "u", "u"]
    gc = pd.crosstab(g["g"], pd.Series(h))
    wc = real_pd.crosstab(w["g"], real_pd.Series(h))
    for col in ("u", "v"):
        assert gc[col].tolist() == wc[col].tolist()
    assert list(gc.index) == list(wc.index)


def test_cut_qcut_match():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    gcut = pd.cut(x, [0, 4, 8], labels=["lo", "hi"]).tolist()
    wcut = list(real_pd.cut(x, [0, 4, 8], labels=["lo", "hi"]))
    assert gcut == [str(v) for v in wcut]
    gq = pd.qcut(x, 2, labels=["l", "h"]).tolist()
    wq = list(real_pd.qcut(x, 2, labels=["l", "h"]))
    assert gq == [str(v) for v in wq]


def test_get_dummies_matches():
    s = ["a", "b", "a", "c"]
    gd = pd.get_dummies(pd.Series(s, name="col"))
    wd = real_pd.get_dummies(real_pd.Series(s, name="col"))
    for c in gd.keys():
        assert gd[c].tolist() == [int(v) for v in wd[c].tolist()]
    gdrop = pd.get_dummies(pd.Series(s, name="col"), drop_first=True)
    wdrop = real_pd.get_dummies(real_pd.Series(s, name="col"),
                                drop_first=True)
    assert sorted(map(str, gdrop.keys())) == \
        sorted(map(str, wdrop.columns))


def test_to_numeric_and_value_counts():
    got = pd.to_numeric(["1", "2.5", "x"], errors="coerce").tolist()
    want = real_pd.to_numeric(["1", "2.5", "x"],
                              errors="coerce").tolist()
    assert got[0] == want[0] and got[1] == want[1]
    assert math.isnan(got[2]) and math.isnan(want[2])
    s = ["a", "b", "a", "a", "c"]
    gv = pd.Series(s).value_counts()
    wv = real_pd.Series(s).value_counts()
    assert gv.tolist() == wv.tolist()
    assert list(gv.index) == list(wv.index)


def test_read_csv_matches(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b,c\n1,2.5,x\n3,NA,y\n5,7.5,z\n")
    g = pd.read_csv(str(p))
    w = real_pd.read_csv(str(p))
    assert list(g.columns) == list(w.columns)
    assert g["a"].tolist() == w["a"].tolist()
    assert g["b"].tolist()[0] == w["b"].tolist()[0]
    assert math.isnan(g["b"].tolist()[1])
    assert g["c"].tolist() == w["c"].tolist()


def test_corr_cov_describe():
    g, w = _pair()
    gc = g[["x", "y"]].corr()
    wc = w[["x", "y"]].corr()
    assert gc["y"].tolist()[0] == pytest.approx(
        wc["y"].tolist()[0], rel=1e-12)
    gv = g[["x", "y"]].cov()
    wv = w[["x", "y"]].cov()
    assert gv["y"].tolist()[0] == pytest.approx(
        wv["y"].tolist()[0], rel=1e-12)
    gd = g.describe()
    wd = w.describe()
    assert gd["x"].tolist() == pytest.approx(wd["x"].tolist())


def test_apply_iterrows_pivot():
    g, w = _pair()
    ga = g.apply(lambda r: r["x"] + 1, axis=1)
    wa = w.apply(lambda r: r["x"] + 1, axis=1)
    assert ga.tolist() == wa.tolist()
    rows_g = [(i, r["x"]) for i, r in g.iterrows()]
    rows_w = [(i, r["x"]) for i, r in w.iterrows()]
    assert rows_g == rows_w
    gp = g.pivot_table(values="x", index="g", columns="g",
                       aggfunc="sum")
    assert gp["a"].tolist()[0] == 9.0  # 1+3+5


def test_str_accessor_and_to_datetime():
    s = ["Foo", "BAR", "baz"]
    assert pd.Series(s).str.lower().tolist() == \
        real_pd.Series(s).str.lower().tolist()
    assert pd.Series(s).str.contains("a").tolist() == \
        real_pd.Series(s).str.contains("a").tolist()
    d = pd.to_datetime(["2024-01-15", "2024-03-02"])
    wd = real_pd.to_datetime(["2024-01-15", "2024-03-02"])
    assert d.dt.year.tolist() == [v for v in wd.year]
    assert d.dt.month.tolist() == [v for v in wd.month]
