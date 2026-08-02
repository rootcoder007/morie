"""morie.fn._frame_core checked against hand-computed expectations.

Every fixture is a six-row frame, so each expected value below is
exact by hand (pandas semantics: ddof=1, NaN-skipping reductions,
linear-interpolation quantiles)."""

import math

import pytest

from morie.fn import _frame_core as pd


DATA = {
    "g": ["a", "b", "a", "b", "a", "c"],
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "y": [2.0, float("nan"), 6.0, 8.0, 10.0, 12.0],
}


def _frame():
    return pd.DataFrame({k: list(v) for k, v in DATA.items()})


def test_series_reductions_match():
    s = pd.Series([1.0, 2.0, float("nan"), 4.0, 10.0])
    # NaN-skipping over [1, 2, 4, 10]
    assert s.sum() == 17.0
    assert s.mean() == 4.25
    assert s.std() == pytest.approx(math.sqrt(16.25), rel=1e-12)
    assert s.var() == pytest.approx(16.25, rel=1e-12)
    assert s.median() == 3.0
    # linear interpolation: pos 0.25*(4-1) = 0.75 between 1 and 2
    assert s.quantile(0.25) == pytest.approx(1.75)
    assert s.count() == 4
    assert s.min() == 1.0 and s.max() == 10.0


def test_series_ops_and_masking():
    s = pd.Series([1.0, 5.0, 3.0])
    assert (s * 2 + 1).tolist() == [3.0, 11.0, 7.0]
    assert s[(s > 2).tolist()].tolist() == [5.0, 3.0]
    assert s.clip(2.0, 4.0).tolist() == [2.0, 4.0, 3.0]
    assert s.rank().tolist() == [1.0, 3.0, 2.0]
    assert s.diff().tolist()[1:] == [4.0, -2.0]
    assert s.cumsum().tolist() == [1.0, 6.0, 9.0]


def test_dataframe_basics_match():
    g = _frame()
    assert g.shape == (6, 3)
    assert list(g.columns) == ["g", "x", "y"]
    assert g["x"].mean() == 3.5
    assert g.dropna().shape == (5, 3)
    assert g.fillna(0.0)["y"].sum() == 38.0
    gm = g[(g["x"] > 2.5).tolist()]
    assert gm.shape == (4, 3)
    assert gm["y"].tolist()[0] == 6.0


def test_sort_values_and_iloc():
    g = _frame()
    gs = g.sort_values("x", ascending=False)
    assert gs["x"].tolist() == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    assert g.iloc[2]["x"] == 3.0
    assert g.iloc[slice(1, 4)]["x"].tolist() == [2.0, 3.0, 4.0]


def test_groupby_matches():
    g = _frame()
    gm = g.groupby("g")["x"].mean()
    # groups (sorted): a -> [1, 3, 5], b -> [2, 4], c -> [6]
    assert gm.tolist() == [3.0, 3.0, 6.0]
    assert list(gm.index) == ["a", "b", "c"]
    gs = g.groupby("g").sum()
    assert gs["x"].tolist() == [9.0, 6.0, 6.0]
    # y sums skip the NaN in group b: a=2+6+10, b=8, c=12
    assert gs["y"].tolist() == [18.0, 8.0, 12.0]
    ga = g.groupby("g").agg({"x": "mean", "y": "max"})
    assert ga["x"].tolist() == [3.0, 3.0, 6.0]
    assert ga["y"].tolist() == [10.0, 8.0, 12.0]


def test_merge_matches():
    left_d = {"k": ["a", "b", "c"], "v": [1, 2, 3]}
    right_d = {"k": ["a", "b", "d"], "u": [10, 20, 40]}
    gl = pd.DataFrame(left_d).merge(pd.DataFrame(right_d), on="k")
    assert gl.shape == (2, 3)
    assert gl["v"].tolist() == [1, 2]
    assert gl["u"].tolist() == [10, 20]
    go = pd.DataFrame(left_d).merge(pd.DataFrame(right_d), on="k",
                                    how="left")
    assert go.shape == (3, 3)
    assert go["u"].tolist()[:2] == [10, 20]
    assert math.isnan(go["u"].tolist()[2])


def test_concat_matches():
    a = {"x": [1, 2], "y": [3, 4]}
    b = {"x": [5], "y": [6]}
    gc = pd.concat([pd.DataFrame(a), pd.DataFrame(b)],
                   ignore_index=True)
    assert gc["x"].tolist() == [1, 2, 5]
    assert gc["y"].tolist() == [3, 4, 6]


def test_crosstab_matches():
    g = _frame()
    h = ["u", "v", "u", "v", "u", "u"]
    gc = pd.crosstab(g["g"], pd.Series(h))
    # pairs: (a,u) x3, (b,v) x2, (c,u) x1
    assert gc["u"].tolist() == [3, 0, 1]
    assert gc["v"].tolist() == [0, 2, 0]
    assert list(gc.index) == ["a", "b", "c"]


def test_cut_qcut_match():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # right-closed bins (0, 4], (4, 8]
    assert pd.cut(x, [0, 4, 8], labels=["lo", "hi"]).tolist() == \
        ["lo", "lo", "lo", "lo", "hi", "hi", "hi", "hi"]
    # median split at 4.5
    assert pd.qcut(x, 2, labels=["l", "h"]).tolist() == \
        ["l", "l", "l", "l", "h", "h", "h", "h"]


def test_get_dummies_matches():
    s = ["a", "b", "a", "c"]
    gd = pd.get_dummies(pd.Series(s, name="col"))
    want = {"a": [1, 0, 1, 0], "b": [0, 1, 0, 0], "c": [0, 0, 0, 1]}
    for c in gd.keys():
        assert gd[c].tolist() == want[str(c)]
    gdrop = pd.get_dummies(pd.Series(s, name="col"), drop_first=True)
    assert sorted(map(str, gdrop.keys())) == ["b", "c"]


def test_to_numeric_and_value_counts():
    got = pd.to_numeric(["1", "2.5", "x"], errors="coerce").tolist()
    assert got[0] == 1.0 and got[1] == 2.5 and math.isnan(got[2])
    gv = pd.Series(["a", "b", "a", "a", "c"]).value_counts()
    assert gv.tolist() == [3, 1, 1]
    assert list(gv.index) == ["a", "b", "c"]


def test_read_csv_matches(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b,c\n1,2.5,x\n3,NA,y\n5,7.5,z\n")
    g = pd.read_csv(str(p))
    assert list(g.columns) == ["a", "b", "c"]
    assert g["a"].tolist() == [1, 3, 5]
    assert g["b"].tolist()[0] == 2.5
    assert math.isnan(g["b"].tolist()[1])
    assert g["c"].tolist() == ["x", "y", "z"]


def test_corr_cov_describe():
    g = _frame()
    # pairwise NaN drop leaves y = 2x exactly: corr 1, cov 2*var(x)
    gc = g[["x", "y"]].corr()
    assert gc["y"].tolist()[0] == pytest.approx(1.0, rel=1e-12)
    gv = g[["x", "y"]].cov()
    # remaining x = [1,3,4,5,6]: var (ddof=1) = 14.8/4 = 3.7
    assert gv["y"].tolist()[0] == pytest.approx(7.4, rel=1e-12)
    gd = g.describe()
    assert gd["x"].tolist() == pytest.approx(
        [6.0, 3.5, math.sqrt(3.5), 1.0, 2.25, 3.5, 4.75, 6.0])


def test_apply_iterrows_pivot():
    g = _frame()
    ga = g.apply(lambda r: r["x"] + 1, axis=1)
    assert ga.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    rows_g = [(i, r["x"]) for i, r in g.iterrows()]
    assert rows_g == [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0),
                      (4, 5.0), (5, 6.0)]
    gp = g.pivot_table(values="x", index="g", columns="g",
                       aggfunc="sum")
    assert gp["a"].tolist()[0] == 9.0  # 1+3+5


def test_str_accessor_and_to_datetime():
    s = ["Foo", "BAR", "baz"]
    assert pd.Series(s).str.lower().tolist() == ["foo", "bar", "baz"]
    assert pd.Series(s).str.contains("a").tolist() == \
        [False, False, True]
    d = pd.to_datetime(["2024-01-15", "2024-03-02"])
    assert d.dt.year.tolist() == [2024, 2024]
    assert d.dt.month.tolist() == [1, 3]
