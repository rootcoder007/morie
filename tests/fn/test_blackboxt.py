import math

import pytest

from morie.fn.blackboxt import blackbox_transpose_fit


def _data():
    rows = []
    for i in range(40):
        t1 = math.sin(1.3 * i)
        t2 = math.cos(0.7 * i + 0.4)
        rows.append(
            [round(4 + t1 * (j - 3) * 0.6 + t2 * math.sin(j + 1) + 0.3 * math.sin(i * j + 1)) for j in range(7)]
        )
    rows[3][2] = None
    rows[10][5] = None
    rows[17][0] = float("nan")
    return rows


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and v != 9


def _r2(pairs):
    n = len(pairs)
    sa = math.fsum(a for a, _ in pairs)
    sb = math.fsum(b for _, b in pairs)
    saa = math.fsum(a * a for a, _ in pairs)
    sbb = math.fsum(b * b for _, b in pairs)
    sab = math.fsum(a * b for a, b in pairs)
    return (n * sab - sa * sb) ** 2 / ((n * saa - sa * sa) * (n * sbb - sb * sb))


def test_one_dimension_recomputes_fit_from_the_returned_parameters():
    rows = _data()
    r = blackbox_transpose_fit(rows, dims=1)
    s1 = r.fits[0]["singular"]
    coords = [s[1] for s in r.stimuli[0]]
    assert math.fsum(c * c for c in coords) == pytest.approx(1.0, abs=1e-12)
    assert max(coords, key=abs) > 0
    sse = []
    by_stim = [[] for _ in range(7)]
    for j, (c, w, r2) in enumerate(r.individuals[0]):
        pairs = [(c + math.sqrt(s1) * coords[i] * w, rows[j][i]) for i in range(7) if _ok(rows[j][i])]
        assert r2 == pytest.approx(_r2(pairs), abs=1e-9)
        sse += [(a - b) ** 2 for a, b in pairs]
        for i in range(7):
            if _ok(rows[j][i]):
                by_stim[i].append((c + math.sqrt(s1) * coords[i] * w, rows[j][i]))
    assert r.fits[0]["SSE"] == pytest.approx(math.fsum(sse), rel=1e-10)
    for i in range(7):
        assert r.stimuli[0][i][0] == len(by_stim[i])
        assert r.stimuli[0][i][2] == pytest.approx(_r2(by_stim[i]), abs=1e-9)
    vals = [v for row in rows for v in row if _ok(v)]
    assert r.n_data == len(vals) and r.n_miss == 3 and r.n_row == 7 and r.n_col == 40
    assert r.ss_mean == pytest.approx(math.fsum(v * v for v in vals) - math.fsum(vals) ** 2 / len(vals), rel=1e-12)
    f = r.fits[0]
    assert f["SSE_explained"] == pytest.approx(r.ss_mean - f["SSE"], rel=1e-12)
    assert f["percent"] == pytest.approx(100 * f["SSE_explained"] / r.ss_mean, rel=1e-12)


def test_two_dimensions_orthonormal_and_nested_fits():
    r = blackbox_transpose_fit(_data(), dims=2)
    C = [s[1:3] for s in r.stimuli[1]]
    for a in range(2):
        for b in range(2):
            assert math.fsum(row[a] * row[b] for row in C) == pytest.approx(1.0 if a == b else 0.0, abs=1e-12)
    assert r.fits[1]["SSE"] < r.fits[0]["SSE"]
    assert r.fits[0]["percent"] + r.fits[1]["percent"] == pytest.approx(r.fits[1]["cumulative_percent"], rel=1e-12)
    assert all(0 <= s[3] <= 1 for s in r.stimuli[1])


def test_missing_codes_and_dropped_respondents():
    rows = _data()
    coded = [[9 if not _ok(v) else v for v in row] for row in rows]
    a = blackbox_transpose_fit(rows, dims=1)
    b = blackbox_transpose_fit(coded, missing=[9], dims=1)
    assert a.stimuli == b.stimuli
    rows[5] = [None, None, None, None, None, 3, 4]
    c = blackbox_transpose_fit(rows, dims=1)
    assert c.individuals[0][5] is None and c.n_col == 39
    with pytest.raises(ValueError):
        blackbox_transpose_fit(rows[:6], dims=1)
    with pytest.raises(ValueError):
        blackbox_transpose_fit(rows, dims=0)
