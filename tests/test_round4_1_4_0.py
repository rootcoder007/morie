"""Round-4 findings of the 1.4.0 fresh-user test agent (l14, 2026-10-03), each with its fix."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_drops_the_placeholder_tests_after_the_src_reinclude():
    text = (ROOT / "pyproject.toml").read_text()
    block = text[text.index("sdist.exclude = [") :]
    block = block[: block.index("]")]
    # gitignore semantics: the last matching pattern wins, so the exclusion must follow "!/src/morie/**"
    assert block.index('"/src/morie/tests/"') > block.index('"!/src/morie/**"')


def test_cihi_picks_the_sheet_with_data_over_a_wide_instructions_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from morie.ingest.cihi import _read_xlsx_streaming

    wb = openpyxl.Workbook()
    ins = wb.active
    ins.title = "Instructions"
    ins["A1"] = "How to read these tables"
    ins.cell(row=16, column=16383, value=" ")  # declared dimension 16 x 16,383, one paragraph of text
    t1 = wb.create_sheet("Table 1")
    t1.append(["province", "year", "count", "rate"])
    for i in range(12):
        t1.append([f"P{i}", 2024, i * 10, i / 10])
    path = tmp_path / "cihi.xlsx"
    wb.save(path)
    df = _read_xlsx_streaming(str(path))
    assert list(df.columns)[:2] == ["province", "year"]
    assert len(df) == 12


def test_stratified_sample_writes_the_design_weight():
    from morie.fn import _frame_core as pd
    from morie.sampling import stratified_sample

    df = pd.DataFrame({"s": ["A"] * 40 + ["B"] * 10, "x": list(range(50))})
    out = stratified_sample(df, "s", 5, seed=1)
    w = dict(zip(out["s"], out[".weight"]))
    assert w == {"A": 40 / 5, "B": 10 / 5}
    assert math.isclose(sum(out[".weight"]), 50.0)


def test_current_locale_reports_the_code_in_effect(monkeypatch):
    from morie import i18n

    monkeypatch.setenv("MORIE_LOCALE", "es-MX")
    assert i18n.current_locale() == "es"
    monkeypatch.setenv("MORIE_LOCALE", "xx")
    assert i18n.current_locale() == "en"


def test_own_file_route_names_the_path_under_morie_data_dir():
    from morie import data

    route = data.dataset_route(data.DATASET_CATALOG["mapq"])
    assert route == "own file: $MORIE_DATA_DIR/datasets/vsr/TKARONTOMAPQ.xlsx"


def test_power_two_proportion_rows_match_the_r_route_columns_and_formulas():
    from morie.fn import _frame_core as pd
    from morie.modules import _two_proportion_rows

    df = pd.DataFrame(
        {
            "gender": [1] * 6 + [2] * 4,
            "y": [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
            "weight": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    (row,) = _two_proportion_rows(df, "y", 1, 2)
    p1 = (1 * 1 + 1 * 1) / 8
    p2 = (1 + 1 + 2) / 6
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    nd = NormalDist()
    z_a, z_b = nd.inv_cdf(0.975), nd.inv_cdf(0.8)
    w = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    deff = len(w) * sum(x * x for x in w) / sum(w) ** 2
    n_eq = 2 * ((z_a + z_b) / abs(h)) ** 2
    se = math.sqrt(1 / 6 + 1 / 4)
    assert (row["group1"], row["group2"]) == ("Female", "Male")
    assert math.isclose(row["p1"], p1) and math.isclose(row["p2"], p2) and math.isclose(row["h"], h)
    assert math.isclose(row["n_eq"], n_eq)
    assert math.isclose(row["n_eq_eff"], n_eq * deff)
    assert math.isclose(row["power_srs"], nd.cdf(abs(h) / se - z_a))
    assert math.isclose(row["power_deff"], nd.cdf(abs(h) / (se * math.sqrt(deff)) - z_a))
    assert row["power_scope"] == "y" and row["analysis_mode"] == "observational"


def test_siu_report_page_reads_the_body_without_menus_or_contents(monkeypatch):
    from morie.ingest import siu

    page = b"""<html><body><nav>Media Centre News Releases</nav>
    <div class="header-body">Public Reports Director's Reports</div>
    <div class="siu-content"><h2>SIU Director's Report - Case # 17-OVI-201</h2>
    <h3>Contents:</h3><ul><li><a href="#m">Mandate of the SIU</a></li>
    <li><a href="#d">Analysis and Director's Decision</a></li></ul>
    <h2 id="m">Mandate of the SIU</h2><p>The SIU investigates.</p>
    <h2>Evidence</h2><p>Two witnesses.</p>
    <h2 id="d">Analysis and Director's Decision</h2><p>There are no grounds to charge the officer.</p>
    </div></body></html>"""

    class _Resp:
        status_code = 200
        content = page
        headers = {"content-type": "text/html"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _Resp()

    monkeypatch.setattr(siu.httpx, "Client", _Client)
    text = siu.fetch_report_text("https://example.invalid/report")
    assert "Media Centre" not in text and "Public Reports" not in text
    assert text.count("Analysis and Director's Decision") == 1
    fields = siu.extract_report_fields(text)
    assert fields["report_id"] == "17-OVI-201"
    assert fields["conclusion"] == "There are no grounds to charge the officer."
    assert fields["sections"]["evidence"] == "Two witnesses."


def test_promax_matches_r_stats_promax_on_a_fixed_loading_matrix():
    from morie.psymet import _inv, _mm, _promax, _tr

    L = [[0.7, 0.2], [0.65, 0.25], [0.6, 0.3], [0.3, 0.6], [0.25, 0.65], [0.2, 0.7]]
    P, U = _promax(L)
    # stats::promax(L, m = 4) in R 4.5, %.15g
    ref = [
        0.772451091716417,
        0.688045061080567,
        0.603639030444717,
        0.0972028466296164,
        0.0127968159937663,
        -0.0716092146420838,
        -0.0716092146420839,
        0.0127968159937662,
        0.0972028466296164,
        0.603639030444717,
        0.688045061080567,
        0.772451091716417,
    ]
    got = [P[i][j] for j in range(2) for i in range(6)]  # column-major, as R prints
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(got, ref))
    phi = _inv(_mm(_tr(U), U))
    assert math.isclose(phi[0][1], 0.649092173759171, abs_tol=1e-9)


def test_omega_hierarchical_separates_group_factors():
    from morie.fn import _array_core as np
    from morie.psymet import mcdo

    rng = np.random.default_rng(4)
    n = 400
    g = [rng.normal() for _ in range(n)]
    s1 = [rng.normal() for _ in range(n)]
    s2 = [rng.normal() for _ in range(n)]
    X = [
        [0.5 * g[r] + 0.6 * s1[r] + 0.6 * rng.normal() for _ in range(4)]
        + [0.5 * g[r] + 0.6 * s2[r] + 0.6 * rng.normal() for _ in range(4)]
        for r in range(n)
    ]
    o1 = mcdo(X, nf=1)
    assert math.isclose(o1.hier, o1.total, abs_tol=1e-3)  # one factor is its own general factor
    o2 = mcdo(X, nf=2)
    assert o2.hier < 0.7 < o2.total  # two group factors carry real variance


def test_formula_sandbox_refuses_file_io(tmp_path):
    from morie._safe_expr import safe_eval_expr
    from morie.fn import _array_core as np

    ns = {"np": np, "x": np.array([1.0, 2.0])}
    target = (tmp_path / "pwned").as_posix()
    for expr in (
        f'np.savez("{target}", x)',
        f'np.savez_compressed("{target}", x)',
        'np.load("/etc/hostname")',
        f'x.tofile("{target}")',
        f'x.dump("{target}")',
        f'np.array(x).to_csv("{target}")',
    ):
        with pytest.raises(ValueError, match="not allowed"):
            safe_eval_expr(expr, ns)
    assert list(tmp_path.iterdir()) == []
    # arithmetic still works, conversions included
    assert safe_eval_expr("np.sin(x[0]) + np.array(x).tolist()[1]", ns) == pytest.approx(math.sin(1.0) + 2.0)
    assert safe_eval_expr('"ab".upper()', {}) == "AB"


def test_stage_lines_count_and_can_be_silenced(monkeypatch):
    import io

    from morie._progress import Stages

    buf = io.StringIO()
    monkeypatch.delenv("MORIE_NO_PROGRESS", raising=False)
    st = Stages("logistic-models", 2, stream=buf)
    st.step("first")
    st.step("second")
    lines = buf.getvalue().splitlines()
    assert lines[0].startswith("logistic-models [1/2] first (") and lines[1].startswith(
        "logistic-models [2/2] second ("
    )
    monkeypatch.setenv("MORIE_NO_PROGRESS", "1")
    quiet = io.StringIO()
    Stages("x", 1, stream=quiet).step("y")
    assert quiet.getvalue() == ""


def test_verify_finds_a_module_folder_inside_the_directory_given(tmp_path):
    from morie.inspector import MODULE_SPECS, verify_directory

    mod = "power-design"
    first = MODULE_SPECS[mod].output_files[0]
    (tmp_path / mod).mkdir()
    (tmp_path / mod / first).write_text("metric,value\nanalysis_n,10\n")
    reports = verify_directory(tmp_path, module_name=mod)
    assert [Path(r.file_path).name for r in reports] == [first]
    # without --module, CSVs in subfolders are found when the folder itself holds none
    assert len(verify_directory(tmp_path)) == 1


def test_interactive_status_says_whether_the_layer_was_verified(tmp_path, monkeypatch):
    from morie import _interactive as ia

    lines = []
    d = tmp_path / "layer"
    d.mkdir()
    for f in ia.FILES:
        (d / f).write_text("# x\n")
    (d / "VERSION").write_text(ia.package_version() + "\n")
    monkeypatch.setattr(ia, "data_dir", lambda: d)
    monkeypatch.setattr(ia, "FILES", ia.FILES)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(ia, "__file__", str(pkg / "_interactive.py"))
    ia.status(out=lines.append)
    assert any("NOT verified" in x for x in lines)
    (d / "VERIFIED").write_text("ok\n")
    lines.clear()
    ia.status(out=lines.append)
    assert any("verified against the bundled manifest" in x for x in lines)


def test_relay_reports_the_registry_size():
    from morie.fn._registry import REGISTRY
    from morie.perseus_relay import _registry_size

    assert _registry_size() == len(REGISTRY) > 5710


def test_compiled_reports_carry_the_authors_not_a_placeholder_line():
    from morie.reporting import Report, compile_report

    rep = Report(title="T", authors=["Smith, J.", "Doe, A."], date="2026-10-03")
    md = compile_report(rep, output_format="markdown")
    tex = compile_report(rep, output_format="latex")
    html = compile_report(rep, output_format="html")
    assert "**Authors**: Smith, J., Doe, A." in md
    assert "\\author{Smith, J. \\and Doe, A.}" in tex
    assert "<p><strong>Authors</strong>: Smith, J., Doe, A.</p>" in html
    assert not any("Lao Tzu" in x for x in (md, tex, html))


def test_power_rows_keep_gender_labels_a_frame_already_holds():
    from morie.fn import _frame_core as pd
    from morie.modules import _two_proportion_rows

    df = pd.DataFrame({"gender": ["Female"] * 3 + ["Male"] * 3, "y": [1, 0, 1, 0, 0, 1], "weight": [1.0] * 6})
    (row,) = _two_proportion_rows(df, "y", "Female", "Male")
    assert (row["group1"], row["group2"]) == ("Female", "Male")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell-style $BROWSER entries")
def test_login_browser_open_does_not_wait_for_the_browser(tmp_path, monkeypatch):
    import time

    from morie import hosted

    seen = tmp_path / "seen.txt"
    script = tmp_path / "slow_browser.py"
    script.write_text(
        "import sys, time, pathlib\n" f"pathlib.Path({str(seen)!r}).write_text(sys.argv[1])\n" "time.sleep(30)\n"
    )
    monkeypatch.setenv("BROWSER", f"{sys.executable} {script}")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    t0 = time.monotonic()
    hosted._open_browser("https://github.com/login/device")
    assert time.monotonic() - t0 < 5
    for _ in range(100):
        if seen.exists():
            break
        time.sleep(0.05)
    assert seen.read_text() == "https://github.com/login/device"

    seen.unlink()
    monkeypatch.setenv("BROWSER", f"{sys.executable} {script} --url=%s")
    hosted._open_browser("https://x.invalid/a")
    for _ in range(100):
        if seen.exists():
            break
        time.sleep(0.05)
    assert seen.read_text() == "--url=https://x.invalid/a"


def test_login_browser_open_falls_back_to_webbrowser(monkeypatch):
    from morie import hosted

    class _Gui:
        def open(self, uri):
            opened.append(uri)

    opened = []
    monkeypatch.setattr(hosted, "_can_open_browser", lambda: True)
    monkeypatch.setenv("BROWSER", "/nonexistent/browser-binary")
    monkeypatch.setattr(hosted.webbrowser, "get", lambda: _Gui())
    hosted._open_browser("https://github.com/login/device")
    assert opened == ["https://github.com/login/device"]
    # a console browser (plain GenericBrowser) is never started: it would take over the terminal
    monkeypatch.setattr(hosted.webbrowser, "get", lambda: hosted.webbrowser.GenericBrowser("lynx"))
    monkeypatch.delenv("BROWSER")
    hosted._open_browser("https://github.com/login/device")
    assert opened == ["https://github.com/login/device"]


def test_login_opens_no_browser_headless_or_over_ssh(monkeypatch):
    from morie import hosted

    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    assert hosted._can_open_browser() is False
    monkeypatch.delenv("SSH_CONNECTION")
    monkeypatch.delenv("SSH_TTY", raising=False)
    if sys.platform.startswith("linux"):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert hosted._can_open_browser() is False
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert hosted._can_open_browser() is True
    called = []
    monkeypatch.setattr(hosted, "_can_open_browser", lambda: False)
    monkeypatch.setattr(hosted.webbrowser, "get", lambda: called.append(1))
    hosted._open_browser("https://github.com/login/device")
    assert called == []
