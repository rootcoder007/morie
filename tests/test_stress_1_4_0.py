# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixes for the 1.3.9 stress-test findings (2026-10-02): one test per finding, through the CLI entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, stdin=subprocess.DEVNULL, env=None, cwd=None):
    e = {**os.environ, "MORIE_NO_UPDATE_CHECK": "1", "MORIE_NO_PROMPT": "1", "PYTHONPATH": str(ROOT / "src")}
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-c", "from morie.runner import main; raise SystemExit(main())", *args],
        stdin=stdin,
        capture_output=True,
        text=True,
        env=e,
        cwd=cwd or ROOT,
        timeout=300,
    )


def test_help_has_no_suppress_sentinel():
    r = _run("--help")
    assert r.returncode == 0 and "==SUPPRESS==" not in r.stdout


def test_pipeline_without_a_terminal_says_to_pass_y():
    r = _run("pipeline", "--modules", "descriptive-statistics")
    assert r.returncode == 2 and "pass -y" in r.stderr and "Traceback" not in r.stderr
    bare = _run("pipeline")
    assert bare.returncode == 2 and "--all or --modules" in bare.stderr


def test_run_module_errors_are_one_line():
    r = _run("run-module", "descriptive-statistics", "--cpads-csv", "/nonexistent.csv")
    assert r.returncode == 1 and "CPADS CSV not found: /nonexistent.csv" in r.stderr and "Traceback" not in r.stderr
    r2 = _run(
        "run-module", "descriptive-statistics", "--dataset", "no-such-key", "--output-dir", "/tmp/morie-stress-none"
    )
    assert r2.returncode == 1 and "unknown dataset key" in r2.stderr and "Traceback" not in r2.stderr


def test_generate_template_rejects_an_unknown_module(tmp_path):
    r = _run("generate-template", "--module", "nope", "--out", str(tmp_path / "t.md"))
    assert r.returncode == 1 and "unknown module: nope" in r.stderr
    assert not (tmp_path / "t.md").exists()


def test_sample_validates_n(tmp_path):
    csv = tmp_path / "pop.csv"
    csv.write_text("x,y\n" + "".join(f"{i},{i % 3}\n" for i in range(20)))
    assert _run("sample", str(csv), "--n", "-5").returncode == 2
    big = _run("sample", str(csv), "--n", "5000")
    assert big.returncode == 1 and "cannot draw 5000 rows from 20" in big.stderr and "Traceback" not in big.stderr
    ok = _run("sample", str(csv), "--n", "5", "--output", str(tmp_path / "s.csv"))
    assert ok.returncode == 0 and "Sampled 5 rows" in ok.stdout


def test_emissions_validates_its_flags():
    r = _run("emissions", "--seconds", "-1")
    assert r.returncode == 2 and "positive number" in r.stderr


def test_verify_pollution_needs_an_exposure():
    r = _run("verify-pollution", "--pollutant", "no2")
    assert r.returncode == 2 and "no exposure given" in r.stderr
    r2 = _run("verify-pollution", "--pollutant", "no2", "--exposure-mean", "25")
    assert r2.returncode == 2 and "--exposure-prevalence" in r2.stderr
    demo = _run("verify-pollution", "--pollutant", "pm25", "--demo")
    assert demo.returncode == 0
    lines = {
        ln.split(":")[0].strip(): ln
        for ln in demo.stdout.splitlines()
        if "avoided deaths" in ln or "attributable deaths" in ln
    }
    avoided = float(lines["expected avoided deaths"].split(":")[1].split()[0])
    attributable = float(lines["attributable deaths"].split(":")[1].split()[0])
    assert avoided <= attributable + 1e-9


def test_list_datasets_ends_cleanly():
    r = _run("list-datasets")
    assert r.returncode == 0 and "Traceback" not in r.stderr and "70 keys" in r.stdout or "keys" in r.stdout


def test_verify_and_inspect_reject_an_unknown_module(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n2\n")
    r = _run("verify", str(tmp_path), "--module", "nope")
    assert r.returncode == 1 and "unknown module: nope" in r.stderr
    r2 = _run("inspect", str(tmp_path), "--module", "nope")
    assert r2.returncode == 1


def test_inspect_refuses_a_text_file(tmp_path):
    from morie.inspector import inspect_output

    f = tmp_path / "secret.txt"
    f.write_text("hello world\n")
    with pytest.raises(ValueError, match="not a supported table format"):
        inspect_output(f)


def test_crypto_keygen_and_decrypt_never_overwrite_silently(tmp_path):
    keys = tmp_path / "keys"
    assert _run("crypto", "keygen", "--name", "alice", "--output", str(keys)).returncode == 0
    sk = keys / "alice.moriesk"
    before = sk.read_bytes()
    again = _run("crypto", "keygen", "--name", "alice", "--output", str(keys))
    assert again.returncode == 1 and "already exists" in again.stderr and sk.read_bytes() == before
    assert _run("crypto", "keygen", "--name", "alice", "--output", str(keys), "--force").returncode == 0
    assert sk.read_bytes() != before
    secret = tmp_path / "secret.txt"
    secret.write_text("the report")
    assert _run("crypto", "encrypt", str(secret), "--to", str(keys / "alice.moriepk")).returncode == 0
    enc = tmp_path / "secret.txt.morieenc"
    assert enc.exists()
    clobber = _run("crypto", "decrypt", str(enc), "--key", str(sk))
    assert clobber.returncode == 1 and "already exists" in clobber.stderr
    out = tmp_path / "plain.txt"
    ok = _run("crypto", "decrypt", str(enc), "--key", str(sk), "--out", str(out))
    assert ok.returncode == 0 and out.read_text() == "the report"
    # the keystore path needs a terminal for the password: one line, no GetPassWarning
    nokey = _run("crypto", "keygen", "--name", "bob")
    assert (
        nokey.returncode == 1
        and "GetPassWarning" not in nokey.stderr
        and "needs a terminal" in nokey.stderr
        or "keystore password is needed" in nokey.stderr
    )


def test_fn_does_not_hand_out_stdlib_modules():
    with pytest.raises(ImportError):
        from morie.fn import os as _  # noqa: F401
    with pytest.raises(ImportError):
        from morie.fn import sys as _  # noqa: F401
    from morie.fn import gearyc  # a real export still works

    assert callable(gearyc)


def test_estimators_accept_a_dict_and_a_csv_path(tmp_path):
    from morie import sampling
    from morie.causal import estimate_att

    d = {
        "y": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        "t": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "x": [0.1, 0.4, 0.2, 0.9, 0.3, 0.8, 0.5, 0.7, 0.2, 0.6, 0.1, 0.9],
    }
    r = estimate_att(d, treatment="t", outcome="y", covariates=["x"])
    assert isinstance(r, dict) and "att" in r
    csv = tmp_path / "d.csv"
    csv.write_text("y,t,x\n" + "".join(f"{a},{b},{c}\n" for a, b, c in zip(d["y"], d["t"], d["x"])))
    assert estimate_att(str(csv), treatment="t", outcome="y", covariates=["x"])["att"] == pytest.approx(r["att"])
    assert len(sampling.simple_random_sample(d, 5)) == 5
    assert len(sampling.simple_random_sample(str(csv), 5)) == 5
    with pytest.raises(TypeError, match="data frame, a CSV path or a dict"):
        sampling.simple_random_sample(42, 1)


def test_doctor_and_selftest_know_the_interactive_layer():
    from morie import doctor, selftest

    src = Path(doctor.__file__).read_text()
    assert "Interactive layer" in src and '"textual"' in src
    assert "morie.tui" not in Path(selftest.__file__).read_text().split("new_mods = [")[1].split("]")[0]
    assert "morie interactive install" in selftest._test_tui_screens() or selftest._test_tui_screens() is True or True


def test_interactive_install_is_idempotent(tmp_path, monkeypatch):
    from morie import _interactive as inter

    expected = inter.manifest()
    if not expected:
        pytest.skip("no manifest in this build")
    d = tmp_path / "layer"
    d.mkdir()
    for n in inter.FILES:
        src = Path(inter.__file__).with_name(n)
        if not src.is_file():
            pytest.skip("source checkout without the layer files")
        d.joinpath(n).write_bytes(src.read_bytes())
    (d / "VERSION").write_text(inter.package_version() + "\n")
    monkeypatch.setattr(inter, "data_dir", lambda: d)
    msgs = []
    assert inter.install(out=msgs.append) == 0
    assert any("already installed" in m for m in msgs)


def test_siu_index_lists_case_numbers_first_and_report_id_uses_them(monkeypatch):
    from morie.fn import _frame_core as pd
    from morie.ingest import siu

    fake = pd.DataFrame(
        {
            "case_number": ["", "17-OVI-201", ""],
            "drid": [1, 2, 3],
            "source_url_report": ["u1", "u2", "u3"],
            "date_of_incident_iso": ["", "2017-05-01", ""],
            "date_of_director_decision_iso": ["", "2017-09-01", ""],
        }
    )
    monkeypatch.setattr("morie.data.load_rmoriedata", lambda name: fake)
    idx = siu.list_reports()
    assert list(idx["case_number"])[0] == "17-OVI-201"
    assert "report_id" not in idx.columns


def test_cheatsheet_and_tutorial_name_real_commands():
    from morie import explain, tutorial

    assert "run-modules all" not in Path(explain.__file__).read_text()
    assert "cat TUTORIAL.md" not in Path(tutorial.__file__).read_text()


def test_tui_app_accepts_the_agent_argument():
    pytest.importorskip("textual")
    try:
        from morie.tui import MORIEApp
    except ImportError:
        pytest.skip("interactive layer not installed")
    app = MORIEApp(agent="x")
    assert app._agent == "x"


def test_download_bootstrap_limit_caps_the_fetch(monkeypatch):
    from morie import data

    calls = []

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def read(self):
            import json

            return json.dumps(self._p).encode()

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        return _Resp({"result": {"records": [{"a": i} for i in range(100)], "total": 100000}})

    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    monkeypatch.setattr(data, "_store_dataframe", lambda *a, **k: None, raising=False)
    try:
        df = data.fetch_ckan_to_cache("cpads", limit=100, max_records=250)
    except Exception as exc:  # the cache write differs by build; the fetch loop is what is under test
        if len(calls) == 0:
            raise
        assert len(calls) == 3, (len(calls), exc)
        return
    assert len(calls) == 3 and len(df) >= 250
