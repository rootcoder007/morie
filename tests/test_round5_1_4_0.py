"""Fresh-user test, round 5 (morie 1.4.0 sdist on l14, 2026-10-03): each finding with its fix."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import sys
import zipfile

import pytest


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership and permission bits")
def test_fn_loader_ignores_a_zip_planted_in_a_world_writable_cache(tmp_path, monkeypatch):
    import morie.fn as fn

    fn_dir = tmp_path / "pkg"
    fn_dir.mkdir()
    xz = fn_dir / "_fnsrc.json.xz"
    with lzma.open(xz, "wb") as fh:
        fh.write(json.dumps({"zzpc": "SOURCE = 'genuine'\n"}).encode())
    tag = hashlib.sha256(xz.read_bytes()).hexdigest()[:12]

    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)  # what another user could pre-create in /tmp
    planted = shared / f"fnsrc-{tag}.zip"
    with zipfile.ZipFile(planted, "w") as zf:
        zf.writestr("zzpc.py", "SOURCE = 'planted'\n")
    private = tmp_path / "private"

    monkeypatch.setattr(fn, "_FN_DIR", str(fn_dir))
    monkeypatch.setattr(fn, "_candidate_cache_dirs", lambda: [str(shared), str(private)])
    monkeypatch.setattr(fn, "__path__", [str(fn_dir)])
    fn._install_fnsrc()

    assert str(planted) not in fn.__path__
    built = private / f"fnsrc-{tag}.zip"
    assert str(built) in fn.__path__
    assert zipfile.ZipFile(built).read("zzpc.py").decode() == "SOURCE = 'genuine'\n"
    assert not (private.stat().st_mode & 0o077)  # the directory it made is private

    # a world-writable zip in our own directory is not trusted either
    built.chmod(0o666)
    assert fn._owned_private(str(built), False) is False
    assert fn._owned_private(str(private), True) is True
    link = tmp_path / "link"
    link.symlink_to(private)
    assert fn._owned_private(str(link), True) is False


def test_interactive_layer_edited_after_install_is_not_loaded(tmp_path, monkeypatch):
    import shutil

    from morie import _interactive as ia

    src = os.path.dirname(ia.__file__)
    if not all(os.path.isfile(os.path.join(src, n)) for n in ia.FILES):
        pytest.skip("needs the five layer files of a source checkout")
    d = tmp_path / "layer"
    d.mkdir()
    for n in ia.FILES:
        shutil.copy(os.path.join(src, n), d / n)
    version = ia.package_version()
    (d / "VERSION").write_text(version + "\n")
    (d / "VERIFIED").write_text("sha256 against the bundled manifest\n")
    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(d))

    path: list[str] = []
    assert ia.changed_files(d) == []
    assert ia.activate(path, version) is True and str(d) in path

    with open(d / "polyglot.py", "a", encoding="utf-8") as fh:
        fh.write('\nprint("PWNED")\n')
    path = []
    assert ia.changed_files(d) == ["polyglot.py"]
    assert ia.activate(path, version) is False and path == []
    lines: list[str] = []
    pkg = tmp_path / "site" / "morie"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(ia, "__file__", str(pkg / "_interactive.py"))  # an installed wheel, not the checkout
    assert ia.status(out=lines.append) == 1
    assert any("NOT loaded: polyglot.py changed" in x for x in lines)

    # a --no-verify install is the user's choice and is not re-checked
    (d / "VERIFIED").unlink()
    assert ia.changed_files(d) == []


def test_formula_sandbox_refuses_memmap_and_runaway_arithmetic(tmp_path):
    from morie._safe_expr import safe_eval_expr

    np = pytest.importorskip("numpy")
    victim = tmp_path / "victim.txt"
    victim.write_text("SECRET\n")
    path = victim.as_posix()
    for expr in (
        f'np.memmap("{path}", mode="r", dtype="uint8")',
        f'np.memmap("{path}", mode="w+", shape=(3,), dtype="uint8")',
    ):
        with pytest.raises(ValueError, match="not allowed"):
            safe_eval_expr(expr, {"np": np})
    assert victim.read_text() == "SECRET\n"  # neither read nor truncated

    with pytest.raises(ValueError, match="too large"):
        safe_eval_expr("10**10**10", {})
    with pytest.raises(ValueError, match="too large"):
        safe_eval_expr('"a" * 10**8', {})
    with pytest.raises(ValueError, match="too large"):
        safe_eval_expr("[0] * 10**7", {})

    # ordinary formulas are untouched
    assert safe_eval_expr("2**10 + 3*4", {}) == 1036
    assert safe_eval_expr('"ab" * 3', {}) == "ababab"
    assert safe_eval_expr("(-2)**3", {}) == -8
    assert safe_eval_expr("2**0.5", {}) == pytest.approx(2**0.5, abs=1e-15)
    x = np.array([1.0, 2.0, 3.0])
    assert list(safe_eval_expr("x**2 + 1", {"x": x})) == [2.0, 5.0, 10.0]


def test_stratified_total_n_with_a_rounding_correction_draws_exactly_n():
    from morie.sampling import stratified_sample

    data = {"g": ["a"] * 3 + ["b"] * 3 + ["c"] * 4, "y": list(range(10))}
    # 5 * (0.3, 0.3, 0.4) = 1.5, 1.5, 2.0 -> rounds to 2, 2, 2 = 6: the largest stratum gives one back
    out = stratified_sample(data, "g", 5, proportional=True, seed=1)
    rows = out.to_dict("records") if hasattr(out, "to_dict") else out
    assert len(rows) == 5
    by = {}
    for r in rows:
        by.setdefault(r["g"], []).append(r[".weight"])
    assert sum(len(v) for v in by.values()) == 5
    sizes = {"a": 3, "b": 3, "c": 4}
    for g, w in by.items():  # each row stands for N_h / n_h units
        assert all(x == pytest.approx(sizes[g] / len(w), abs=1e-12) for x in w)
    assert sum(sum(v) for v in by.values()) == pytest.approx(10.0, abs=1e-12)


def test_boolean_eval_leading_not_and_lower_case_names():
    from morie.fn import boolean_eval

    assert boolean_eval("~A", {"A": 1}).value == 0
    assert boolean_eval("~A", {"A": 0}).value == 1
    assert boolean_eval("~(A & B)", {"A": 1, "B": 1}).value == 0
    assert boolean_eval("A & (B | ~C)", {"A": 1, "B": 0, "C": 0}).value == 1
    assert boolean_eval("A & (B | ~C)", {"A": 1, "B": 0, "C": 1}).value == 0
    assert boolean_eval("A ^ B", {"A": 1, "B": 0}).value == 1
    assert boolean_eval("A ^ B", {"A": 1, "B": 1}).value == 0
    assert boolean_eval("o | n", {"o": 0, "n": 1}).value == 1
    assert boolean_eval("o & n", {"o": 0, "n": 1}).value == 0
    with pytest.raises(ValueError, match="Undefined variable: D"):
        boolean_eval("A & D", {"A": 1})


def _verify_pollution(*args: str):
    import subprocess

    return subprocess.run(
        [sys.executable, "-m", "morie.runner", "verify-pollution", *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("outcome", ["ihd", "stroke", "all_cause_mortality"])
def test_verify_pollution_one_outcome_throughout(outcome):
    r = _verify_pollution(
        "--pollutant", "pm25", "--outcome", outcome, "--exposure-mean", "12", "--exposure-prevalence", "1", "--json"
    )
    assert r.returncode == 0, r.stderr[-500:]
    rep = json.loads(r.stdout[r.stdout.index("{") :])
    pipe = rep["pipeline"]
    # the burden stage uses the same relative risk as the concentration-response stage
    assert pipe["burden"]["extra"]["rr"] == pytest.approx(pipe["crf"]["rr"], rel=1e-9)
    assert pipe["burden"]["paf"] == pytest.approx(pipe["paf"], rel=1e-12)
    # with prevalence 1, Levin's PAF = 1 - 1/RR = the displaced share: avoided never exceeds attributable
    attributable = pipe["burden"]["attributable_cases"]
    assert pipe["displaced"]["deaths_displaced"] <= attributable * (1 + 1e-9)


def test_verify_pollution_unknown_outcome_and_infinite_inputs():
    r = _verify_pollution(
        "--pollutant", "pm25", "--outcome", "nonsense", "--exposure-mean", "12", "--exposure-prevalence", "1"
    )
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert "--outcome nonsense" in r.stderr and "ihd" in r.stderr and "stroke" in r.stderr
    r = _verify_pollution(
        "--pollutant", "no2", "--outcome", "lung_cancer", "--exposure-mean", "30", "--exposure-prevalence", "1"
    )
    assert r.returncode == 2 and "respiratory" in r.stderr and "Traceback" not in r.stderr
    for mean, rate in (("inf", "500"), ("30", "inf")):
        r = _verify_pollution(
            "--pollutant",
            "no2",
            "--exposure-mean",
            mean,
            "--baseline-rate",
            rate,
            "--exposure-prevalence",
            "1",
            "--json",
        )
        assert r.returncode == 1, (mean, rate, r.stdout[-300:])
        assert json.loads(r.stdout[r.stdout.index("{") :])["status"] == "assumption_failure"


def test_emissions_unknown_country_is_labelled_world_average(tmp_path, monkeypatch):
    from morie.emissions import known_country_codes, run_check, summary_text

    monkeypatch.setenv("MORIE_EMISSIONS_OFFLINE", "1")  # the unknown code falls back to detection; keep it offline

    if not known_country_codes():
        pytest.skip("energy-mix table not bundled")
    data = run_check(0.2, str(tmp_path / "em"), capsule=False, country_iso_code="XYZ")
    assert data.country_iso_code == ""
    assert "(XYZ)" not in summary_text(data, None)
    assert "world average" in summary_text(data, None)
    ok = run_check(0.2, str(tmp_path / "em2"), capsule=False, country_iso_code="can")
    assert ok.country_iso_code == "CAN"


def test_ebac_counts_a_standard_drink_as_0_6_oz_of_ethanol():
    from morie import calculate_ebac
    from morie.fn.ebac import calculate_ebac as fn_ebac

    # 5 standard drinks, 150 lb man, 2 h: Widmark US units, A = 5 x 0.6 fl oz
    expected = 5 * 0.6 * 5.14 / (150 * 0.73) - 0.015 * 2
    assert expected == pytest.approx(0.11082, abs=1e-5)
    assert calculate_ebac(5, 150, 2, 0.73) == pytest.approx(expected, abs=1e-12)
    assert fn_ebac(5, 150, 2, 0.73) == pytest.approx(expected, abs=1e-12)
    assert calculate_ebac(5, 150, 2, 0.66) == pytest.approx(5 * 0.6 * 5.14 / (150 * 0.66) - 0.03, abs=1e-12)


def _lcg_frame(n=600, seed=20261003):
    from morie.fn import _frame_core as fpd

    s, u = seed, []
    for _ in range(n * 10):
        s = (1664525 * s + 1013904223) % (2**32)
        u.append(s / 2**32)
    c = [u[k * n : (k + 1) * n] for k in range(10)]

    def pick(col, values):
        return [values[int(v * len(values)) % len(values)] for v in col]

    return fpd.DataFrame(
        {
            "weight": [0.6 + v for v in c[0]],
            "alcohol_past12m": [1] * n,
            "heavy_drinking_30d": [int(v < 0.45) for v in c[1]],
            "ebac_tot": [0.2 * v for v in c[2]],
            "ebac_legal": [int(v < 0.4) for v in c[3]],
            "cannabis_any_use": [int(v < 0.35) for v in c[4]],
            "age_group": pick(c[5], [1, 2, 3, 4]),
            "gender": pick(c[6], [1, 2, 3]),
            "province_region": pick(c[7], [1, 2, 3, 4, 5]),
            "mental_health": pick(c[8], [1, 2, 3, 4, 5]),
            "physical_health": pick(c[9], [1, 2, 3, 4, 5]),
        }
    )


@pytest.mark.parametrize("where", ["morie.investigation", "morie.fn.wlog"])
def test_logistic_models_enter_survey_codes_as_categories(where):
    import importlib

    mod = importlib.import_module(where)
    out = mod.run_weighted_logistic_analysis(_lcg_frame())
    terms = list(out["logistic_odds_ratios"]["term"])
    # one odds ratio per non-reference level, not one "per unit of code", named by label as the R route does
    assert sum(t.startswith("province_region_label") for t in terms) == 4
    assert {t for t in terms if t.startswith("gender_label")} == {"gender_labelMale", "gender_labelNon-binary"}
    assert sum(t.startswith("age_group_label") for t in terms) == 3
    assert "province_region" not in terms and "gender" not in terms
    int_terms = list(out["logistic_interaction_odds_ratios"]["term"])
    assert sum("cannabis_any_use:gender_label" in t for t in int_terms) == 2
    assert out["logistic_interaction_tests"]["df_num"].tolist() == [2]


def _morie(*args: str, cwd=None, env=None):
    import subprocess

    return subprocess.run(
        [sys.executable, "-m", "morie.runner", *args], capture_output=True, text=True, timeout=300, cwd=cwd, env=env
    )


def test_crypto_names_a_swapped_secret_or_public_key(tmp_path):
    keys = tmp_path / "k"
    assert _morie("crypto", "keygen", "--name", "alice", "--output", str(keys)).returncode == 0
    msg = tmp_path / "msg.txt"
    msg.write_text("hello\n")
    sk, pk = keys / "alice.moriesk", keys / "alice.moriepk"
    r = _morie("crypto", "encrypt", str(msg), "--to", str(sk), "--out", str(tmp_path / "x.enc"))
    assert r.returncode == 1 and "is a secret key; encrypt to the public key" in r.stderr
    assert "1.3.x" not in r.stderr and not (tmp_path / "x.enc").exists()
    assert _morie("crypto", "encrypt", str(msg), "--to", str(pk), "--out", str(tmp_path / "ok.enc")).returncode == 0
    r = _morie("crypto", "decrypt", str(tmp_path / "ok.enc"), "--key", str(pk), "--out", str(tmp_path / "back.txt"))
    assert r.returncode == 1 and "is a public key; decrypt needs the secret key" in r.stderr
    assert "1.3.x" not in r.stderr


def test_cli_usage_errors_are_one_line(tmp_path):
    r = _morie("exec", "")
    assert r.returncode == 2 and "No code provided" in r.stderr
    r = _morie("sample", str(tmp_path / "nofile.csv"), "--n", "3")
    assert r.returncode == 1 and "Traceback" not in r.stderr and "file not found" in r.stderr


def test_edit_run_names_itself_when_the_layer_is_missing(monkeypatch, capsys):
    import argparse

    from morie import runner

    seen = []
    monkeypatch.setitem(sys.modules, "morie._exec_guard", None)  # as in the published wheel
    monkeypatch.setattr(runner, "_add_interactive_layer", lambda what: seen.append(what) or False)
    ns = argparse.Namespace(
        code=None, filename=None, lang="python", exec_file="e.py", co_dir=None, verb="'morie edit --run'"
    )
    assert runner._handle_exec(ns) == 1
    assert seen == ["'morie edit --run'"]


def test_mcdo_validates_the_number_of_factors():
    from morie.fn import _array_core as anp
    from morie.psymet import mcdo

    rows = [[((i * 7 + j * 3) % 11) / 11 + (i % 5) / 5 for j in range(6)] for i in range(120)]
    x = anp.asarray(rows, dtype=anp.float64)
    for bad in (-1, 0, 6, 7, 1.5, True):
        with pytest.raises(ValueError, match="nf must be a whole number of factors between 1 and 5"):
            mcdo(x, nf=bad)


def test_published_files_do_not_point_at_files_missing_from_the_sdist():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "WHATS_NEW.md"):
        text = (root / name).read_text(encoding="utf-8")
        rel = [t for t in re.findall(r"\]\(([^)#]+)\)", text) if not re.match(r"[a-z]+:", t)]
        assert rel == [], (name, rel)
    py = (root / "pyproject.toml").read_text(encoding="utf-8")
    for pat in ('"morie/*.c"', '"morie/*.h"', '"morie/Makefile"', '"morie/Modelfile*"'):
        assert pat in py
    poly = root / "src" / "morie" / "polyglot.py"
    if poly.is_file():
        assert "brew install" not in poly.read_text(encoding="utf-8")


def _report():
    import morie.reporting as r
    from morie.fn import _frame_core as fpd

    rep = r.Report(title="Test & <Stuff>", authors=["Ada", "Emile"], date="2026-10-03")
    rep.add_section(r.ReportSection("Methods", "We used 50% of $data_1 with x_i & y^2 and <b>html</b>."))
    rep.add_table("Odds & ends", fpd.DataFrame({"term": ["a_b", "c"], "OR": [1.23456, 12345.0], "n": [3, None]}))
    return r, rep


def test_reports_render_tables_without_pandas_and_escape_text(tmp_path):
    r, rep = _report()
    md = r.compile_report(rep)
    assert "| term | OR | n |" in md and "| a_b | 1.23456 | 3 |" in md and "| c | 12345 |  |" in md
    assert "**Authors**: Ada, Emile\n\n**Date**: 2026-10-03" in md
    tex = r.compile_report(rep, output_format="latex")
    assert r"\title{Test \& <Stuff>}" in tex
    assert r"We used 50\% of \$data\_1 with x\_i \& y\textasciicircum{}2" in tex
    assert r"a\_b & 1.23456 & 3 \\" in tex and r"\caption{Table 1. Odds \& ends}" in tex
    page = r.compile_report(rep, output_format="html")
    assert "<h1>Test &amp; &lt;Stuff&gt;</h1>" in page
    assert "&lt;b&gt;html&lt;/b&gt;" in page and "<b>html</b>" not in page
    assert "<td>a_b</td><td>1.23456</td><td>3</td>" in page  # tables are part of the HTML now
    for fmt in ("docx", "txt", "nonsense"):
        with pytest.raises(ValueError, match="output_format must be one of markdown, latex, html"):
            r.compile_report(rep, output_format=fmt)
    for name in ("r.docx", "r.pdf", "r.xyz"):
        with pytest.raises(ValueError, match="cannot tell the format"):
            r.save_report(rep, tmp_path / name)
        assert not (tmp_path / name).exists()
    assert r.save_report(rep, tmp_path / "r.tex").read_text(encoding="utf-8").startswith(r"\title{")


def test_format_p_value_refuses_impossible_p():
    from morie.reporting import format_p_value

    assert format_p_value(0.034) == "p = .034"
    assert format_p_value(0.0002) == "p < .001"
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            format_p_value(bad)


def test_verify_pollution_reads_a_naps_pull_in_ppb(tmp_path):
    naps = tmp_path / "naps.csv"
    naps.write_text("station_id,datetime_local,value,unit\n1,a,10,ppb\n1,b,20,ppb\n2,a,15,ppb\n")
    r = _verify_pollution("--pollutant", "no2", "--exposure-csv", str(naps), "--json")
    assert r.returncode == 0, r.stderr[-300:]
    assert "converted to ug/m3" in r.stderr
    rep = json.loads(r.stdout[r.stdout.index("{") :])
    assert rep["inputs"]["exposure_mean"] == pytest.approx(15 * 1.88, abs=1e-9)
    bad = tmp_path / "bad.csv"
    bad.write_text("value,unit\n3,ppm\n")
    r = _verify_pollution("--pollutant", "pm25", "--exposure-csv", str(bad))
    assert r.returncode == 2 and "is not ug/m3" in r.stderr


def test_a_limited_ckan_fetch_is_never_cached(monkeypatch):
    import morie.data as data

    stored = []
    rows = [{"a": i} for i in range(5)]
    monkeypatch.setattr(data, "_ckan_source", lambda key: ({"resource_id": "rid", "metadata_url": ""}, "tbl", False))
    monkeypatch.setattr(
        data, "_urlopen_json_with_retry", lambda url, timeout, **_kw: {"result": {"records": rows, "total": len(rows)}}
    )
    monkeypatch.setattr(data, "cache_store", lambda df, name, db=None, **_kw: stored.append((name, len(df))))
    preview = data.fetch_ckan_to_cache("ocs22bt", max_records=3)
    assert len(preview) >= 3 and stored == []
    data.fetch_ckan_to_cache("ocs22bt")
    assert stored == [("tbl", 5)]


@pytest.mark.parametrize(
    ("portal", "args", "error", "expect"),
    [
        ("tps", ["--layer", "robbery"], "layer returned zero features: where='OCC_YEAR=2099'", "zero features"),
        ("ckan", ["--package", "no-such-pkg"], 'package_show -> HTTP 404: {"help": "x"}', "no-such-pkg was not found"),
    ],
)
def test_ingest_portal_errors_are_one_line(monkeypatch, capsys, portal, args, error, expect):
    import importlib

    from morie import runner

    mod = importlib.import_module(f"morie.ingest.{portal}")
    cls = type("PortalError", (RuntimeError,), {"__module__": mod.__name__})

    def boom(argv):
        raise cls(error)

    monkeypatch.setattr(mod, "cli", boom)
    monkeypatch.setattr(sys, "argv", ["morie", "ingest", portal, *args])
    assert runner.main() == 1
    err = capsys.readouterr().err
    assert expect in err and "Traceback" not in err and err.count("\n") == 1


_SIU_PAGE = """<html><body><div class="col-lg-9 siu-content">
<h2>SIU Director’s Report - Case # 17-OVI-201</h2>
<h4>Warning:</h4><p>This page contains graphic content that can shock, offend and upset.</p>
<h3>Contents:</h3><ul><li><a href="#mandate">Mandate of the SIU</a></li><li><a href="#inv">The Investigation</a></li></ul>
<div><strong>News Releases for this Case:</strong><ul><li><a href="/news/1">SIU Investigation Started</a></li></ul>
<strong>French:</strong><ul><li><a href="/fr/46">Director's Report for Case # 17-OVI-201.</a></li></ul></div>
<h2 id="mandate">Mandate of the SIU</h2><div>The Unit investigates police incidents.</div>
<h2>Information restrictions</h2>
<p>Witness statements gathered in the course of the investigation.</p>
<p>Pursuant to <abbr>PHIPA</abbr>, any information related to health is withheld.</p>
<h2 id="inv">The Investigation</h2>
<h3>Notification of the SIU</h3>
<p>At 11:46 a.m. on August 3rd, 2017, the Guelph Police Service (<abbr>GPS</abbr>) notified the SIU.</p>
<h2>Incident Narrative</h2><p>Three men attempted to rob a bank.</p>
<h2>Analysis and Director's Decision</h2><p>There are no grounds to proceed with charges.</p>
</div></body></html>"""


def test_siu_report_page_text_has_no_chrome_and_real_sections():
    pytest.importorskip("bs4")
    from bs4 import BeautifulSoup

    from morie.ingest import siu

    body = BeautifulSoup(_SIU_PAGE, "html.parser").find("div", class_="siu-content")
    text = siu._report_page_text(body)
    lines = text.splitlines()
    assert lines[:2] == ["SIU Director’s Report - Case # 17-OVI-201", "Mandate of the SIU"]
    for chrome in ("Warning", "graphic content", "Contents", "News Releases", "French"):
        assert chrome not in text
    assert "the Guelph Police Service (GPS) notified the SIU." in text  # inline tags stay in the sentence
    assert "Pursuant to PHIPA, any information" in text
    f = siu.extract_report_fields(text)
    assert f["sections"]["investigation"].startswith("Notification of the SIU")
    assert "PHIPA" not in f["sections"]["investigation"]
    assert f["sections"]["narrative"] == "Three men attempted to rob a bank."
    assert f["conclusion"] == "There are no grounds to proceed with charges."


def test_describe_resolves_public_names_through_their_module():
    from morie.fn import describe
    from morie.fn.describe import _lazy_map

    lazy = _lazy_map()
    differing = [n for n, m in sorted(lazy.items()) if n != m and n[0].islower()][:40]
    assert len(differing) >= 20
    for name in differing:
        assert f"describe({name!r}) -- " in str(describe(name))  # warnings may come first
    head = str(describe("kamath_ch9_fom_loss")).splitlines()[0]
    assert head.startswith("describe('kamath_ch9_fom_loss') -- ") and "# morie.fn" not in head


def test_richresult_from_a_bare_mapping_shows_its_fields():
    import morie.fn as fn

    w = [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ]
    r = fn.morani([1, 2, 3, 4, 10, 2], w)
    text = repr(r)
    assert text and str(r) == text
    for key in r:
        assert key in text


def test_pipeline_checks_module_names_and_keeps_its_tables(tmp_path):
    r = _morie("pipeline", "--modules", "nosuch", "-y", cwd=tmp_path)
    assert r.returncode == 2 and "unknown module: nosuch" in r.stderr
    assert list(tmp_path.iterdir()) == []  # nothing ran, nothing written
    r = _morie("pipeline", "--modules", "power-design", "-y", "--no-carbon", cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-300:]
    assert "morie-output/" in r.stdout
    assert len(list((tmp_path / "morie-output").glob("*.csv"))) >= 10


def test_nist_rds_sends_searchphrase_first_and_pages_by_page(monkeypatch):
    import morie.ingest.forensics as fx

    calls = []
    data = [{"@id": f"ark:{i}", "title": f"t{i}"} for i in range(5)]

    class Resp:
        status_code = 200

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params):
            calls.append(list(params.items()))
            page = params["page"]
            return Resp({"ResultCount": 5, "ResultData": data[(page - 1) * 2 : page * 2]})

    monkeypatch.setattr(fx.httpx, "Client", Client)
    df = fx.fetch_nist_rds(query="cocaine", page_size=2)
    assert len(df) == 5
    assert all(c[0] == ("searchphrase", "cocaine") for c in calls)
    assert [dict(c)["page"] for c in calls] == [1, 2, 3]
    assert all("from" not in dict(c) for c in calls)


def test_no_quote_spliced_into_fn_docstrings_or_values():
    import re
    from pathlib import Path

    fn_dir = Path(__file__).resolve().parents[1] / "src" / "morie" / "fn"
    if not (fn_dir / "amhst.py").is_file():
        pytest.skip("needs the loose fn modules of a source checkout")
    who = (
        r"(Aristotle|Lao Tzu|Confucius|Seneca|Marcus Aurelius|Socrates|Heraclitus|Pythagoras|Euclid|Benjamin Franklin)"
    )
    spliced = re.compile(
        r'^\s*(r?"""[^"\n]*-- ' + who + r'[^"\n]*"""|\'.*-- ' + who + r".*'|test_name=.*-- " + who + r")"
    )
    hits = []
    for p in fn_dir.glob("*.py"):
        if p.name == "_registry.py":  # its fifth field is the epigraph, on purpose
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if spliced.match(line):
                hits.append(f"{p.name}:{i}")
    assert hits == [], hits[:10]
    from morie.fn import red_pill_test

    assert red_pill_test([0.1, 0.4, -0.2, 0.3, 0.2]).test_name == "One-sample t-test (red pill / blue pill)"


def test_doctor_lists_every_trust_knob_once(monkeypatch):
    """One row per knob (a second table described MORIE_TRUST_CHECKPOINT differently)."""
    from morie import doctor

    monkeypatch.setenv("MORIE_NO_EXEC", "1")
    monkeypatch.delenv("MORIE_TRUST_CHECKPOINT", raising=False)
    rows = [c for c in doctor.run_checks()["checks"] if c["label"].startswith("trust: ")]
    names = [c["label"][len("trust: ") :] for c in rows]
    assert sorted(names) == sorted(
        {"MORIE_NO_EXEC", "MORIE_TRUST_CHECKPOINT", "MORIE_ALLOW_REMOTE_INSTALL", "MORIE_ALLOW_RC", "MORIE_ALLOW_CRON"}
    )
    detail = {n: c["detail"] for n, c in zip(names, rows)}
    assert detail["MORIE_NO_EXEC"].startswith("ENABLED")
    assert "tensors and plain containers only" in detail["MORIE_TRUST_CHECKPOINT"]
    assert not hasattr(doctor, "_render_trust_knobs")


def test_smote_sensitivity_is_skipped_on_a_balanced_outcome():
    from morie.investigation import run_weighted_logistic_analysis

    out = run_weighted_logistic_analysis(_lcg_frame())  # heavy drinking ~45/55: ratio >= 0.8
    status = out["logistic_smote_status"].to_dict("records")[0]
    assert status["method"].startswith("skipped: classes already balanced")
    assert len(out["logistic_smote_odds_ratios"]) == 0


def test_r_backed_module_fails_fast_without_r(monkeypatch):
    import time

    import morie.modules as m

    monkeypatch.setattr(m, "_rscript_bin", lambda: None)
    monkeypatch.setattr(m, "_R_READY", None)
    called = []
    monkeypatch.setattr(m, "_cpads_csv_for_run", lambda *a, **k: called.append(1) or "x.csv")
    r_only = next(n for n in m.MODULE_SPECS if n not in m._PY_FALLBACK_MODULES)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="Rscript is not available"):
        m.run_module(r_only, dataset_key="ocp21")
    assert called == [] and time.monotonic() - t0 < 2  # the data were never loaded


def test_doctor_heading_and_pull_summary_are_translated(monkeypatch):
    from morie.i18n import t

    monkeypatch.setenv("MORIE_LOCALE", "fr")
    assert t("doctor.heading").startswith("MORIE Doctor") and "environnement" in t("doctor.heading")
    assert t("pull.wrote_n_rows", path="x.csv", rows=35, cols=15) == "écrit x.csv  (35 lignes, 15 colonnes)"


def test_aipw_picks_the_outcome_model_and_refuses_logistic_on_a_continuous_outcome():
    from morie.causal import estimate_aipw
    from morie.fn import _frame_core as fpd

    n = 200
    u = _lcg_frame(n).to_dict("records")
    rows = [
        {
            "t": r["cannabis_any_use"],
            "z": r["ebac_tot"],
            "y": 1.5 + 0.8 * r["cannabis_any_use"] + 2 * r["ebac_tot"] + r["weight"],
        }
        for r in u
    ]
    d = fpd.DataFrame(rows)
    res = estimate_aipw(d, treatment="t", outcome="y", covariates=["z"])
    assert res["ate"] == pytest.approx(0.8, abs=0.35)
    with pytest.raises(ValueError, match="needs a 0/1 outcome"):
        estimate_aipw(d, treatment="t", outcome="y", covariates=["z"], outcome_model="logistic")


def test_inspect_reads_json(tmp_path):
    import json as _json

    from morie.inspector import inspect_output

    p = tmp_path / "t.json"
    p.write_text(_json.dumps([{"a": 1, "b": 2.5}, {"a": 2, "b": 3.5}]))
    rep = inspect_output(str(p))
    assert rep is not None and "not a supported" not in str(rep)


def test_mapq_without_your_file_is_the_synthetic_panel(tmp_path, monkeypatch):
    import warnings as _w

    from morie.data import load_dataset, synthetic_mapq_panel

    monkeypatch.setenv("MORIE_DATA_DIR", str(tmp_path))
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        df = load_dataset("mapq")
    assert any("synthetic toy panel" in str(w.message) for w in rec)
    assert len(df) == 400 and "ks_score" in df.columns and "EE1" in df.columns
    again = synthetic_mapq_panel()
    assert df.to_dict("records")[:5] == again.to_dict("records")[:5]  # deterministic
    items = [c for c in df.columns if c[:2] in ("EE", "EA", "UA", "ER") and c[2:].isdigit()]
    assert len(items) == 20 and all(1 <= v <= 5 for c in items for v in df[c])
    r = _morie("list-datasets", env={**os.environ, "MORIE_DATA_DIR": str(tmp_path)})
    assert any(line.startswith("mapq") and "synthetic" in line for line in r.stdout.splitlines())


def test_encrypt_to_a_keystore_name_needs_no_password(tmp_path):
    from morie.crypto.keystore import create_keystore, load_public_key, store_keypair

    ks = str(tmp_path / "ks.json")
    create_keystore("pw", path=ks)
    store_keypair("carol", b"\x01" * 1184, b"\x02" * 2400, "pw", path=ks)
    assert load_public_key("carol", path=ks) == b"\x01" * 1184
    with pytest.raises(KeyError):
        load_public_key("nobody", path=ks)


def test_chat_and_ask_share_the_provider_chain_with_the_hosted_tier(monkeypatch):
    import morie.llm as llm

    monkeypatch.setattr(llm, "_hosted_attempt", lambda model: ("https://llm.example.invalid/v1", model or "m", "k"))
    for prov in ("hosted", "ollama"):
        bases = [a[0] for a in llm._provider_attempts(prov, None)]
        assert "https://llm.example.invalid/v1" in bases, prov
    seen = []

    def fake(base_url, model, messages, **kw):
        seen.append((base_url, messages[0]["content"]))
        return "PONG"

    monkeypatch.setattr(llm, "_completion_text", fake)
    assert llm.ask_multi([{"role": "user", "content": "hi"}], provider="hosted") == "PONG"
    assert seen[0][0] == "https://llm.example.invalid/v1"


def test_a_model_not_on_the_key_is_one_clear_error(monkeypatch):
    import httpx

    import morie.llm as llm

    base = "https://llm.example.invalid/v1"
    monkeypatch.setattr(llm, "_hosted_attempt", lambda model: (base, model or "m", "k"))
    req = httpx.Request("POST", base + "/chat/completions")

    def refuse(*a, **k):
        raise httpx.HTTPStatusError("403", request=req, response=httpx.Response(403, request=req))

    monkeypatch.setattr(llm, "_completion_text", refuse)
    with pytest.raises(llm.ModelNotOnKeyError, match="not available on your hosted key"):
        llm.ask("hi", provider="hosted", model="nosuch-model")
    from morie.perseus import ask_percy

    monkeypatch.setattr(llm, "detect_available_provider", lambda: "hosted")
    import morie.perseus as pz

    monkeypatch.setattr(pz, "detect_available_provider", lambda: "hosted")
    payload = ask_percy("hi", model="nosuch-model", stream=False)
    assert payload["failed"] and "not available on your hosted key" in payload["output_text"]


def test_a_custom_system_prompt_keeps_the_module_list():
    from morie.llm import _build_messages, build_morie_context

    msgs = _build_messages("which module?", context=build_morie_context(), system_prompt="You are Perseus.")
    sysmsg = msgs[0]["content"]
    assert sysmsg.startswith("You are Perseus.") and "treatment-effects" in sysmsg and "recommend only these" in sysmsg


def test_r_object_pull_is_saved_not_failed(tmp_path, monkeypatch, capsys):
    import morie.data as data
    from morie import runner

    rdata = tmp_path / "env.RData"
    rdata.write_bytes(b"RDX3\n")

    def fake_load(key, *a, **k):
        raise data.RObjectSavedError(key, rdata)

    monkeypatch.setattr(runner, "load_dataset", fake_load, raising=False)
    monkeypatch.setattr(data, "load_dataset", fake_load)
    out = tmp_path / "otis.out"
    monkeypatch.setattr(sys, "argv", ["morie", "pull", "otis", "--out", str(out)])
    assert runner.main() == 0
    err = capsys.readouterr().err
    assert "is an R object; saved" in err and "rmorie::morie_load_dataset('otis')" in err
    assert (tmp_path / "otis.out.RData").read_bytes() == b"RDX3\n"


def test_unknown_hosted_key_is_named(monkeypatch):
    import morie.datahub as hub

    monkeypatch.setattr(hub, "hosted_manifest", lambda refresh=False: {"datasets": [{"key": "a/b", "rows": 3}]})
    with pytest.raises(KeyError, match="unknown dataset key 'nosuchdb/x'"):
        hub._check_known("nosuchdb/x")
    hub._check_known("a/b")


def test_hosted_table_streams_to_csv_without_a_frame(tmp_path, monkeypatch):
    import gzip as _gz

    import morie.datahub as hub

    monkeypatch.setattr(hub, "hosted_manifest", lambda refresh=False: {"datasets": [{"key": "big/t", "rows": 3}]})

    def fake_get(path, dest, label, timeout=600):
        with _gz.open(dest, "wt", encoding="utf-8", newline="") as fh:
            fh.write('a,b\n1,"x, y"\n2,z\n3,"multi\nline"\n')
        return 1

    monkeypatch.setattr(hub, "_get_to_file", fake_get)
    rows, cols = hub.hosted_to_csv("big/t", tmp_path / "o.csv")
    assert (rows, cols) == (3, 2)
    assert (tmp_path / "o.csv").read_text().startswith('a,b\n1,"x, y"\n')


def test_whole_number_float_columns_are_written_as_integers():
    from morie.fn import _frame_core as fpd
    from morie.runner import _integral_floats_as_int

    df = fpd.DataFrame({"n": [100.0, None, 0.0], "x": [1.5, 2.0, None], "s": ["a", "b", "c"]})
    _integral_floats_as_int(df)
    assert list(df["n"]) == [100, None, 0] and list(df["x"])[:2] == [1.5, 2.0]
    assert "100.0" not in df.to_csv(index=False)


def test_login_while_signed_in_warns_and_needs_force(monkeypatch, capsys):
    import morie.hosted as hosted
    from morie import runner

    monkeypatch.setattr(hosted, "read_credentials", lambda: {"hosted_key": "sk-x", "hosted_user": "someone"})
    started = []
    monkeypatch.setattr(hosted, "device_login", lambda **k: started.append(1))
    monkeypatch.setattr(runner, "_stdin_is_terminal", lambda: False)
    monkeypatch.setattr(sys, "argv", ["morie", "login", "--no-browser"])
    assert runner.main() == 1 and started == []
    err = capsys.readouterr().err
    assert "signed in as someone" in err and "--force" in err
    monkeypatch.setattr(sys, "argv", ["morie", "login", "--no-browser", "--force"])
    assert runner.main() == 0 and started == [1]


def test_every_module_table_has_an_explanation(tmp_path):
    from morie.explain import describe
    from morie.modules import MODULE_SPECS

    for spec in MODULE_SPECS.values():
        for f in spec.output_files:
            if f.endswith(".csv"):
                assert not describe(f).startswith("No registered explanation"), f
    p = tmp_path / "logistic_odds_ratios.csv"
    p.write_text("term,OR,p_value,weird_col\nx,1.2,0.04,3\n")
    text = describe(str(p))
    assert "odds ratio = exp(coefficient)" in text and "weird_col" in text and "SE" not in text


def test_each_stat_repl_handler_documents_its_own_command():
    from morie.stat_commands import commands_by_category

    for cat in ("Descriptive",):
        cmds = commands_by_category().get(cat, [])
        docs = [c.handler_repl.__doc__ for c in cmds if c.handler_repl is not None and c.handler_repl.__doc__]
        if len(docs) > 1:
            assert len(set(docs)) == len(docs)  # every handler read the loop's last description before
            for c in cmds:
                if c.handler_repl is not None and c.handler_repl.__doc__:
                    assert c.description in c.handler_repl.__doc__


def test_mat_reader_reads_an_empty_char_matrix():
    import struct

    from morie import _mat_reader as mr

    def element(dtype, payload):
        pad = (-len(payload)) % 8
        return struct.pack("<II", dtype, len(payload)) + payload + b"\0" * pad

    flags = element(6, struct.pack("<II", 4, 0))  # miUINT32: mxCHAR class
    dims = element(5, struct.pack("<ii", 0, 0))  # miINT32: 0 x 0
    name = element(1, b"s")  # miINT8
    payload = flags + dims + name  # no data element at all
    got_name, txt = mr._matrix(payload)
    assert got_name == "s" and txt == ""


def test_omega_scores_reverse_keyed_items_the_other_way_round():
    """Reversed items used to cancel the general factor (omega 0.15 where psych gives 0.80)."""
    import random
    import warnings

    from morie.psymet import mcdo

    rng = random.Random(7)
    rows = []
    for _ in range(400):
        f = rng.gauss(0, 1)
        rows.append([0.8 * f + 0.6 * rng.gauss(0, 1) for _ in range(6)])
    flipped = [[-v if j in (1, 4) else v for j, v in enumerate(r)] for r in rows]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = mcdo(rows, nf=1)
        b = mcdo(flipped, nf=1)
    assert abs(a.total - b.total) < 1e-9 and abs(a.hier - b.hier) < 1e-9 and a.total > 0.8
    assert any("reverse-keyed" in str(x.message) and "1, 4" in str(x.message) for x in w)


def test_multi_treatment_matching_refuses_a_continuous_column():
    import random

    from morie.fn import _frame_core as pd
    from morie.matching import match_multi_treatment

    rng = random.Random(1)
    t = [rng.randrange(3) for _ in range(300)]
    x = [rng.gauss(0, 1) for _ in range(300)]
    d = pd.DataFrame({"t": t, "x": x, "y": [a + b + rng.gauss(0, 1) for a, b in zip(t, x)]})
    with pytest.raises(ValueError, match="300 distinct values"):
        match_multi_treatment(d, "y", ["x"])


def test_siu_analyses_count_one_row_per_case_and_clean_sex_categories():
    from morie.fn import _frame_core as pd
    from morie.siu import analyze, native

    assert native.to_iso_date("February 30, 2019") == "" and native.to_iso_date("February 29, 2020") == "2020-02-29"
    d = pd.DataFrame(
        {
            "case_number": ["17-OVI-201", "17-OVI-201", "", "18-TCI-001"],
            "_language": ["fr", "en", "unknown", "en"],
            "police_service": ["Toronto Police Service"] * 4,
        }
    )
    s = analyze._clean(d)
    assert len(s) == 2 and list(s["_language"]) == ["en", "en"]
    cats = [
        analyze._sex(v) for v in ("Male", "boy", "female (Complainant #1) and male (Complainant #2)", "trans female")
    ]
    assert cats == ["male", "male", "multiple persons", "transgender"]
    assert analyze._sex("ual assault complaint text " * 4) == "unknown"


def test_sample_cluster_beyond_the_clusters_and_strata_dropped_are_said_in_words(tmp_path):
    import subprocess
    import sys

    csv = tmp_path / "s.csv"
    csv.write_text("id,g,school\n" + "".join(f"{i},{'ABCDEFG'[i % 7]},{i // 5}\n" for i in range(100)))
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from morie.runner import main; raise SystemExit(main())",
            "sample",
            str(csv),
            "--n",
            "50",
            "--method",
            "cluster",
            "--cluster-col",
            "school",
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert r.returncode == 1 and "school has 20 clusters" in r.stderr and "Traceback" not in r.stderr
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from morie.runner import main; raise SystemExit(main())",
            "sample",
            str(csv),
            "--n",
            "3",
            "--method",
            "stratified",
            "--strata-col",
            "g",
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert r.returncode == 0 and "with no rows" in r.stderr
