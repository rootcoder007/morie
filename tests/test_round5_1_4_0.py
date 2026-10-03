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


def test_emissions_unknown_country_is_labelled_world_average(tmp_path):
    from morie.emissions import known_country_codes, run_check, summary_text

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
    # one odds ratio per non-reference level, not one "per unit of code"
    assert sum(t.startswith("C(province_region)") for t in terms) == 4
    assert sum(t.startswith("C(gender)") for t in terms) == 2
    assert sum(t.startswith("C(age_group)") for t in terms) == 3
    assert "province_region" not in terms and "gender" not in terms
    int_terms = list(out["logistic_interaction_odds_ratios"]["term"])
    assert sum("cannabis_any_use:C(gender)" in t for t in int_terms) == 2
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
        data, "_urlopen_json_with_retry", lambda url, timeout: {"result": {"records": rows, "total": len(rows)}}
    )
    monkeypatch.setattr(data, "cache_store", lambda df, name, db=None: stored.append((name, len(df))))
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
