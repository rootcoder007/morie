"""Fixes for the round-7 fresh-user findings (2026-10-03): one test per finding."""

from __future__ import annotations

import pytest

from morie.selftest import run_selftest


def test_selftest_table_is_not_cut_off_a_terminal(capsys):
    assert run_selftest() == 0
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "Status" in line and "Detail" in line)
    assert header.rstrip().endswith("Time")
    for line in out.splitlines():
        if "TUI screens" in line and "SKIP" in line:
            assert "morie interactive install" in line


def test_own_file_messages_name_a_real_directory_when_morie_data_dir_is_unset(monkeypatch, tmp_path):
    import warnings

    from morie import _datapaths, data

    monkeypatch.delenv("MORIE_DATA_DIR", raising=False)
    monkeypatch.setattr(_datapaths, "_user_data_dir", lambda: tmp_path / "userdata")
    monkeypatch.chdir(tmp_path)
    want = tmp_path / "userdata" / "datasets" / "vsr" / "TKARONTOMAPQ.xlsx"
    assert data.dataset_route(data.DATASET_CATALOG["mapq"]) == f"own file: {want}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data.load_dataset("mapq")
    text = " ".join(str(w.message) for w in caught)
    assert f"your file is not at {want};" in text
    assert "$MORIE_DATA_DIR" not in text


def test_a_big_ckan_table_says_which_step_runs_after_the_download(monkeypatch, tmp_path, capsys):
    from morie import data

    rows = [{"_id": i, "a": i, "b": str(i)} for i in range(5)]
    monkeypatch.setattr(data, "_STAGES_MIN_CELLS", 10)
    monkeypatch.delenv("MORIE_NO_PROGRESS", raising=False)
    monkeypatch.setattr(
        data, "_urlopen_json_with_retry", lambda url, timeout, **_kw: {"result": {"records": rows, "total": len(rows)}}
    )
    key = next(k for k in data.CKAN_DATASETS if data._ckan_source(k)[1] != "cpads" and not data._ckan_source(k)[2])
    monkeypatch.setitem(data.CKAN_DATASETS[key], "resource_id", "rid")
    df = data.fetch_ckan_to_cache(key, db_path=tmp_path / "c.sqlite")
    assert len(df) == 5
    err = capsys.readouterr().err
    assert f"{key} [1/2] building the table (5 rows x 3 columns)" in err
    assert f"{key} [2/2] caching it so the next pull is fast" in err


def _wrapped_header_xlsx(path):
    import zipfile

    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        zf.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{main}"><si><t xml:space="preserve">Number of &#10;sites  (total)</t></si><si><t>Region</t></si></sst>',
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{main}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>1</v></c><c r="B1" t="s"><v>0</v></c></row>'
            '<row r="2"><c r="A2" t="inlineStr"><is><t>East</t></is></c><c r="B2"><v>3</v></c></row>'
            "</sheetData></worksheet>",
        )


def test_workbook_headers_wrapped_inside_a_cell_become_one_line_names(tmp_path):
    from morie import data

    wb = tmp_path / "w.xlsx"
    _wrapped_header_xlsx(wb)
    df = data._xlsx_data_sheet(wb)
    assert list(df.columns) == ["Region", "Number of sites (total)"]


def test_pull_failures_read_as_a_message_not_an_exception_class(monkeypatch, capsys):
    from morie import data, runner

    def boom(*a, **k):
        raise RuntimeError("hibp is served from data.rmorie.com: run morie login")

    monkeypatch.setattr(data, "load_dataset", boom)
    if hasattr(runner, "load_dataset"):
        monkeypatch.setattr(runner, "load_dataset", boom)
    monkeypatch.setattr("sys.argv", ["morie", "pull", "hibp"])
    assert runner.main() == 1
    err = capsys.readouterr().err
    assert "pull hibp: hibp is served from data.rmorie.com: run morie login" in err
    assert "RuntimeError" not in err


def test_a_limit_preview_announces_the_rows_it_fetches(monkeypatch, tmp_path, capsys):
    from morie import data

    rows = [{"_id": i, "a": i} for i in range(10)]
    monkeypatch.delenv("MORIE_NO_PROGRESS", raising=False)
    monkeypatch.setattr(
        data, "_urlopen_json_with_retry", lambda url, timeout, **_kw: {"result": {"records": rows, "total": 61096}}
    )
    key = next(k for k in data.CKAN_DATASETS if not data._ckan_source(k)[2])
    monkeypatch.setitem(data.CKAN_DATASETS[key], "resource_id", "rid")
    df = data.fetch_ckan_to_cache(key, limit=10, max_records=10, db_path=tmp_path / "c.sqlite")
    assert len(df) == 10
    err = capsys.readouterr().err
    assert "61,096" not in err


def test_exec_file_runs_the_users_file_in_place(monkeypatch, tmp_path, capfd):
    import pytest

    pytest.importorskip("morie._exec_guard")
    from morie import runner

    script = tmp_path / "g.py"
    script.write_text('import sys\nprint(__file__)\nprint(sys.argv)\nx = 1\nraise ValueError("boom on line 5")\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["morie", "exec", "--file", "g.py"])
    assert runner.main() == 1
    out, err = capfd.readouterr()
    assert out.splitlines()[:2] == ["g.py", "['g.py']"]
    assert 'File "g.py", line 5' in err
    # the traceback starts at the user's frame: none of the runner's own frames
    assert "runpy" not in err and 'File "<string>", line 1' not in err
    assert err.count("File ") == 1


def test_exec_inline_traceback_reads_like_python_c(monkeypatch, capfd):
    import pytest

    pytest.importorskip("morie._exec_guard")
    from morie import runner

    monkeypatch.setattr("sys.argv", ["morie", "exec", "1/0"])
    assert runner.main() == 1
    err = capfd.readouterr().err
    assert 'File "<string>", line 1, in <module>' in err
    assert "morie_exec.py" not in err and "runpy" not in err
    assert err.rstrip().endswith("ZeroDivisionError: division by zero")


def test_morie_no_progress_zero_keeps_the_bars(monkeypatch):
    from morie._progress import Stages

    for v, on in (("0", True), ("false", True), ("", True), ("1", False), ("yes", False)):
        monkeypatch.setenv("MORIE_NO_PROGRESS", v)
        assert Stages("x", 1).enabled is on, v


def test_array_kernels_return_morie_arrays_so_no_numpy_is_needed():
    import pytest

    from morie import _jit

    if not _jit.is_jit_available():
        pytest.skip("compiled core not built")
    import morie.fast as mf

    out = mf.normal_pdf([0.0, 1.0], 0.0, 1.0)
    assert type(out).__module__.startswith("morie.")
    assert [round(float(v), 4) for v in out] == [0.3989, 0.242]
    w = _jit.trimmed_ipw_weights_jit([1.0, 0.0], [0.5, 0.5], 0.01, 0.99)
    assert [float(v) for v in w] == [2.0, 2.0]
    assert len(_jit.bootstrap_mean_jit([1.0, 2.0, 3.0], 7, 1)) == 7


def test_the_r_bridge_uses_the_r_package_whose_version_matches(monkeypatch):
    import subprocess

    import morie.modules as m
    from morie import __version__

    monkeypatch.setattr(m, "_rscript_bin", lambda: "Rscript")
    monkeypatch.setattr(m, "_R_READY", None)
    monkeypatch.setattr(m, "_R_PACKAGE", None)
    monkeypatch.delenv("MORIE_ALLOW_R_VERSION_MISMATCH", raising=False)
    out = f"rmorie 1.3.4\nmorie {__version__}\n"
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    m._r_route_ready()
    assert m._R_PACKAGE == "morie"
    monkeypatch.setattr(m, "_R_READY", None)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "rmorie 1.3.4\n", ""))
    import pytest

    with pytest.raises(RuntimeError, match="R rmorie 1.3.4 is installed, but this is morie"):
        m._r_route_ready()


def test_a_python_capable_module_takes_the_python_route_when_r_is_another_version(monkeypatch):
    import morie.modules as m

    def mismatch():
        raise RuntimeError(
            "R rmorie 1.3.4 is installed, but this is morie 1.4.0: the R-backed modules need the same version"
        )

    def no_r(*a, **k):
        raise AssertionError("the R arm of another version must not run")

    monkeypatch.setattr(m, "_r_route_ready", mismatch)
    monkeypatch.setattr(m, "_run_r_module", no_r)
    monkeypatch.setattr(m, "_cpads_csv_for_run", lambda *a, **k: "x.csv")
    monkeypatch.setattr(m, "run_power_design_module", lambda csv, output_dir=None: "python route")
    assert m.run_module("power-design") == "python route"


def test_tps_dates_build_from_text_parts_with_month_names():
    from morie.fn import _frame_core as pd
    from morie.tps_stochastic import _date_series

    df = pd.DataFrame(
        {
            "OCC_YEAR": ["2024", "2024", "2024"],
            "OCC_MONTH": ["January", "February", "Smarch"],
            "OCC_DAY": ["1", "29", "3"],
        }
    )
    ts = _date_series(df)
    assert [str(v)[:10] for v in ts.tolist()] == ["2024-01-01", "2024-02-29"]


def test_series_sample_draws_the_rows_dataframe_sample_draws():
    from morie.fn import _frame_core as pd

    df = pd.DataFrame({"x": list(range(20))})
    a = df["x"].sample(n=5, random_state=7)
    b = df.sample(n=5, random_state=7)["x"]
    assert a.tolist() == b.tolist() and list(a.index) == list(b.index)
    assert len(df["x"].sample(frac=0.5, random_state=1)) == 10


def test_native_to_datetime_reads_month_names_parts_and_epochs():
    import datetime as dt

    from morie.fn import _frame_core as pd

    for text in (
        "January 5, 2024",
        "5 January 2024",
        "5th of Jan 2024",
        "2024-January-05",
        "05-Jan-2024",
        "JAN 5 2024",
    ):
        assert pd.to_datetime(text) == dt.datetime(2024, 1, 5), text
    assert pd.to_datetime("2024-Sept-03") == dt.datetime(2024, 9, 3)
    parts = pd.DataFrame({"year": ["2024", "2024"], "month": ["December", "Smarch"], "day": ["31", "2"]})
    got = pd.to_datetime(parts, errors="coerce").tolist()
    assert got[0] == dt.datetime(2024, 12, 31) and got[1] != got[1]
    assert pd.to_datetime([0, 86_400_000], unit="ms").tolist() == [dt.datetime(1970, 1, 1), dt.datetime(1970, 1, 2)]


def test_mann_whitney_exact_p_is_the_exact_two_sided_tail():
    import itertools
    import math
    from collections import Counter

    from morie.fn.gb661 import mwu

    def null_counts(m, n):  # every placement of the m X's among m + n ranks
        c = Counter()
        for pos in itertools.combinations(range(m + n), m):
            ys_before, u, xs = 0, 0, set(pos)
            for i in range(m + n):
                if i in xs:
                    u += ys_before
                else:
                    ys_before += 1
            c[u] += 1
        return c

    r = mwu([1, 2, 3, 4, 5], [4.5, 10, 11, 12])  # U = 1
    c = null_counts(5, 4)
    want = 2 * (c[0] + c[1]) / math.comb(9, 5)
    assert r["statistic"] == 1.0
    assert abs(r["p_value"] - want) < 1e-12 and abs(want - 4 / 126) < 1e-12
    c = null_counts(6, 5)
    x, y = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [1, 2, 3, 6, 7]
    u_obs = sum(a > b for a in x for b in y)  # pairs where X is larger: 12
    r = mwu(x, y)
    lower = sum(v for u, v in c.items() if u <= u_obs) / math.comb(11, 6)
    upper = sum(v for u, v in c.items() if u >= u_obs) / math.comb(11, 6)
    assert r["statistic"] == u_obs and abs(r["p_value"] - min(1.0, 2 * min(lower, upper))) < 1e-12


def test_pull_of_a_missing_own_file_says_where_it_goes_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from morie import runner

    monkeypatch.setenv("MORIE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["morie", "pull", "mapq"])
    assert runner.main() == 1
    err = capsys.readouterr().err
    assert (
        f"mapq is your own research file and it is not at {tmp_path / 'data' / 'datasets' / 'vsr' / 'TKARONTOMAPQ.xlsx'}"
        in err
    )
    assert not (tmp_path / "mapq.csv").exists()


def test_pull_all_needs_yes_off_a_terminal(monkeypatch, capsys):
    from morie import runner

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.argv", ["morie", "pull", "--all"])
    assert runner.main() == 2
    assert "pass -y" in capsys.readouterr().err


def test_exec_takes_c_like_python(monkeypatch, capfd):
    import pytest

    pytest.importorskip("morie._exec_guard")
    from morie import runner

    monkeypatch.setattr("sys.argv", ["morie", "exec", "-c", "print(6 * 7)"])
    assert runner.main() == 0
    assert capfd.readouterr().out.strip().endswith("42")


def test_the_tutorial_ends_on_its_last_line(monkeypatch, capsys, tmp_path):
    from morie import tutorial

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)  # the tutorial refuses without a terminal
    monkeypatch.setattr(tutorial, "_prompt", lambda: print("<PROMPT>") or "s")
    tutorial.run()
    out = capsys.readouterr().out.rstrip()
    assert out.endswith("Welcome aboard.")


def test_read_csv_infers_one_type_per_column_as_pandas_does(tmp_path):
    from morie.fn import _frame_core as pd

    f = tmp_path / "t.csv"
    f.write_text("a,b,c,d\n100,1,x,1\n45.3,2,7,\n,3,9,3\n")
    df = pd.read_csv(f)
    a, b, c, d = (df[k].tolist() for k in "abcd")
    assert a[:2] == [100.0, 45.3] and all(isinstance(v, float) for v in a)
    assert b == [1, 2, 3] and all(isinstance(v, int) for v in b)
    assert c == ["x", "7", "9"]
    assert d[0] == 1.0 and d[1] != d[1] and d[2] == 3.0


def test_a_hosted_r_object_is_saved_in_the_data_dir_in_effect(monkeypatch, tmp_path):
    import pytest

    from morie import _datapaths, data, datahub

    old = tmp_path / "userdata"
    key = next(k for k, e in data.DATASET_CATALOG.items() if e.get("hosted_file"))
    entry = data.DATASET_CATALOG[key]
    rel = entry["local_path"].removeprefix("data/")
    (old / rel).parent.mkdir(parents=True)
    (old / rel).write_bytes(b"old copy")
    monkeypatch.setattr(_datapaths, "_user_data_dir", lambda: old)
    monkeypatch.setenv("MORIE_DATA_DIR", str(tmp_path / "chosen"))
    monkeypatch.chdir(tmp_path)
    got = []
    monkeypatch.setattr(datahub, "_get_to_file", lambda path, dest, **k: got.append(dest) or dest.write_bytes(b"new"))
    with pytest.raises(data.RObjectSavedError) as e:
        data.load_dataset(key)
    assert str(tmp_path / "chosen") in str(e.value)
    assert got and str(got[0]).startswith(str(tmp_path / "chosen"))


def test_answers_start_at_their_text(capsys):
    from morie.runner import _drain_stream

    assert _drain_stream(iter(["\n", " \n", "PO", "NG"])) == 4
    assert capsys.readouterr().out == "PONG\n"


def test_doctor_reports_the_trust_knobs_without_the_interactive_layer(monkeypatch):
    import sys

    from morie import doctor

    monkeypatch.setitem(sys.modules, "morie._exec_guard", None)  # the plain wheel has no layer
    monkeypatch.setenv("MORIE_TRUST_CHECKPOINT", "1")
    monkeypatch.setattr(doctor, "_check_morie_version", lambda: (True, "skipped"))
    res = doctor.run_checks()
    rows = [c for c in res["checks"] if c["label"].startswith("trust: ")]
    assert len(rows) == 5
    detail = {c["label"][len("trust: ") :]: c["detail"] for c in rows}
    assert detail["MORIE_TRUST_CHECKPOINT"].startswith("ENABLED -- when set: convert-checkpoint")


def test_r_install_pins_the_r_arm_to_this_version():
    from morie import __version__
    from morie.bricklayer import _r_install_expr

    expr = _r_install_expr()
    assert f"rootcoder007/rmorie@v{__version__}" in expr
    assert f"have() != '{__version__}'" in expr


def test_doctor_fix_without_pip_says_what_to_run(monkeypatch, capsys):
    import importlib.util

    from morie import doctor

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: None if n == "pip" else real(n, *a, **k))
    monkeypatch.setattr("shutil.which", lambda n: None)
    ran = []
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: ran.append(a))
    doctor._heal({"checks": [{"label": "import scipy", "passed": False}]})
    out = capsys.readouterr().out
    assert "no pip; run `uv pip install scipy`" in out and not ran


def test_chat_refuses_an_unknown_agent(monkeypatch, capsys):
    from morie import chat, runner

    monkeypatch.setattr(chat, "list_agents", lambda: [{"name": "analyst", "description": ""}])
    monkeypatch.setattr(chat, "run_chat_repl", lambda agent=None: 0)
    monkeypatch.setattr("sys.argv", ["morie", "chat", "--agent", "nosuch-agent"])
    assert runner.main() == 2
    assert "unknown agent 'nosuch-agent' (known: analyst)" in capsys.readouterr().err


def test_chat_refuses_any_agent_when_no_personas_are_installed(monkeypatch, capsys):
    from morie import chat, runner

    monkeypatch.setattr(chat, "list_agents", lambda: [])
    monkeypatch.setattr(chat, "run_chat_repl", lambda agent=None: 0)
    monkeypatch.setattr("sys.argv", ["morie", "chat", "--agent", "nosuch-agent"])
    assert runner.main() == 2
    assert "this install has no agent personas" in capsys.readouterr().err


def test_percy_local_without_ollama_exits_non_zero(monkeypatch, capsys):
    from pathlib import Path

    from morie import _interactive, runner

    if (
        not _interactive.present(_interactive.data_dir())
        and not Path(runner.__file__).with_name("polyglot.py").is_file()
    ):
        pytest.skip("percy --local runs the interactive layer's agent (morie interactive install)")

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    monkeypatch.setattr("sys.argv", ["morie", "percy", "--local", "--no-stream", "hi"])
    assert runner.main() == 1
    assert "no Ollama answers at http://127.0.0.1:9" in capsys.readouterr().err


def test_convert_checkpoint_names_a_file_that_is_not_one(monkeypatch, tmp_path, capsys):
    from morie import runner

    fake = tmp_path / "fake.pt"
    fake.write_text("notapt\n")
    monkeypatch.setenv("MORIE_TRUST_CHECKPOINT", "1")
    monkeypatch.setattr(
        "sys.argv", ["morie", "convert-checkpoint", "--checkpoint", str(fake), "--output", str(tmp_path / "f.gguf")]
    )
    assert runner.main() == 2
    err = capsys.readouterr().err
    assert "is not a PyTorch checkpoint (a .pt file is a zip archive)" in err and "Traceback" not in err


def test_percysuits_names_an_unreachable_ssh_host(monkeypatch, capsys):
    import subprocess

    from morie import runner

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 255, "", "ssh: connect to host 127.0.0.1 port 22: Connection refused"
        ),
    )
    assert runner._percysuits_get_installed_ssh("nobody@127.0.0.1") == (None, None)
    assert "cannot reach nobody@127.0.0.1 over SSH: ssh: connect to host" in capsys.readouterr().out


def test_login_to_email_needs_an_address(monkeypatch, capsys):
    from morie import runner

    monkeypatch.setattr("sys.argv", ["morie", "login", "--to-email"])
    assert runner.main() == 2
    assert "--to-email needs --email ADDRESS" in capsys.readouterr().err


def test_login_token_reads_a_piped_key(monkeypatch):
    import io

    from morie import hosted, runner

    stored = []
    monkeypatch.setattr(hosted, "store_token", lambda t: stored.append(t))
    monkeypatch.setattr(runner, "_stdin_is_terminal", lambda: False)
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-piped\n"))
    monkeypatch.setattr("sys.argv", ["morie", "login", "--token"])
    assert runner.main() == 0
    assert stored == ["sk-piped"]


def test_pull_reports_no_network_in_the_users_language(monkeypatch, capsys):
    from urllib.error import URLError

    from morie import data, runner

    def offline(*a, **k):
        raise URLError(OSError(101, "Network is unreachable"))

    monkeypatch.setattr(data, "load_dataset", offline)
    if hasattr(runner, "load_dataset"):
        monkeypatch.setattr(runner, "load_dataset", offline)
    monkeypatch.setenv("MORIE_LOCALE", "fr")
    monkeypatch.setattr("sys.argv", ["morie", "pull", "hibub"])
    assert runner.main() == 1
    err = capsys.readouterr().err
    assert "pull hibub: L'appel réseau a échoué : [Errno 101] Network is unreachable" in err


def test_downloads_do_not_retry_without_a_network():
    import errno
    import socket
    from urllib.error import URLError

    from morie._progress import _no_route

    assert _no_route(URLError(OSError(errno.ENETUNREACH, "Network is unreachable")))
    assert _no_route(URLError(socket.gaierror(-2, "Name or service not known")))
    assert not _no_route(ConnectionResetError(104, "reset"))


def test_propensity_scores_writes_the_files_it_declares():
    from morie import causal
    from morie.modules import MODULE_SPECS

    out = causal.run_propensity_ipw_analysis(_cpads_like_frame())
    declared = {f.removesuffix(".csv") for f in MODULE_SPECS["propensity-scores"].output_files}
    assert declared <= set(out)


def _cpads_like_frame(n=300):
    import random

    from morie.fn import _frame_core as pd

    rng = random.Random(7)
    rows = []
    for _ in range(n):
        g = rng.random() < 0.5
        a = rng.choice(["16-19", "20-24", "25+"])
        c = rng.random() < (0.3 + 0.2 * g)
        rows.append(
            {
                "cannabis_any_use": int(c),
                "heavy_drinking_30d": int(rng.random() < 0.2 + 0.2 * c),
                "gender": "Male" if g else "Female",
                "age_group": a,
                "province_region": rng.choice(["Atlantic", "Quebec", "Ontario", "Prairies", "BC"]),
                "mental_health": rng.choice(["Good", "Fair", "Poor"]),
                "physical_health": rng.choice(["Good", "Fair", "Poor"]),
                "weight": 0.5 + rng.random(),
            }
        )
    return pd.DataFrame(rows)


def test_run_modules_skips_r_modules_without_an_r_package_and_runs_the_rest(monkeypatch, capsys):
    import warnings

    from morie import runner

    def fake(name, **k):
        if name == "data-wrangling":
            raise RuntimeError(
                "No R package for the R-backed modules is installed (install rmorie, or morie's R package)"
            )
        warnings.warn("synthetic CPADS frame in use", UserWarning, stacklevel=1)
        return {}

    monkeypatch.setattr(runner, "run_module", fake)
    monkeypatch.setattr("sys.argv", ["morie", "run-modules", "--modules", "data-wrangling", "power-design"])
    assert runner.main() == 0
    out, err = capsys.readouterr()
    assert "data-wrangling skipped (runs in R)" in out and "Completed modules: power-design" in out
    assert "note: synthetic CPADS frame in use" in err and "Traceback" not in err


def test_cli_user_warnings_are_one_note_line(monkeypatch, capsys):
    import warnings

    from morie import runner

    monkeypatch.setattr(warnings, "showwarning", warnings.showwarning)
    runner._cli_warning_notes()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("morie: using the SHIPPED SYNTHETIC CPADS frame", UserWarning, stacklevel=1)
        warnings.warn("morie: using the SHIPPED SYNTHETIC CPADS frame", UserWarning, stacklevel=1)
    err = capsys.readouterr().err
    assert err == "note: morie: using the SHIPPED SYNTHETIC CPADS frame\n"


def test_repl_shows_r_errors_and_honours_the_default_language():
    import shutil

    import pytest

    pytest.importorskip("morie.polyglot")
    if not shutil.which("R"):
        pytest.skip("no R")
    from morie.polyglot import PolyglotEngine, detect_language

    eng = PolyglotEngine(polyglot=False, auto_detect=True)
    try:
        bad = eng._exec_r('stop("boom")')
        assert not bad.success and "boom" in bad.stderr
        ok = eng._exec_r("print(2)")
        assert ok.success and "[1] 2" in ok.stdout
    finally:
        if eng._r_proc:
            eng._r_proc.kill()
    assert detect_language('cat(1+1, "\\n")', "r") == "r"


def test_the_siu_corpus_sex_column_holds_categories():
    from morie.data import _post_load
    from morie.fn import _frame_core as pd

    df = pd.DataFrame(
        {
            "sex_gender_affected": [
                "Male",
                "ual assault. the unit\u2019s jurisdiction covers more than 50 municipal, regional and provincial police services across ontario.",
                "female (Complainant #1) and male (Complainant #2)",
                None,
            ]
        }
    )
    assert _post_load("siu", df)["sex_gender_affected"].tolist() == ["male", "unknown", "multiple persons", None]


def test_every_relative_import_in_the_package_resolves():
    import ast
    import importlib
    import pathlib

    import morie
    from morie._interactive import FILES as layer

    root = pathlib.Path(morie.__file__).parent
    missing = []
    for f in root.rglob("*.py"):
        pkg = ".".join(("morie", *f.relative_to(root).with_suffix("").parts[:-1]))
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                modname = f"{base}.{node.module}" if node.module else base
                try:
                    mod = importlib.import_module(modname)
                except ImportError:
                    continue  # optional extras / the interactive layer
                for a in node.names:
                    if a.name != "*" and not hasattr(mod, a.name):
                        if modname == "morie" and f"{a.name}.py" in layer:
                            continue  # the interactive layer, installed per user by `morie interactive install`
                        try:
                            importlib.import_module(f"{modname}.{a.name}")
                        except ImportError:
                            missing.append(f"{f.relative_to(root)}:{node.lineno} {modname}.{a.name}")
    assert missing == []


def test_an_unknown_option_shows_the_verbs_usage(monkeypatch, capsys):
    import pytest

    from morie import runner

    for argv, prog in (
        (["list-modules", "--nosuchflag-xyz"], "morie list-modules"),
        (["doctor", "--nope"], "morie doctor"),
    ):
        monkeypatch.setattr("sys.argv", ["morie", *argv])
        with pytest.raises(SystemExit) as e:
            runner.main()
        assert e.value.code == 2
        err = capsys.readouterr().err
        assert err.startswith(f"usage: {prog}") and "unrecognized arguments" in err and "{pipeline" not in err


def test_ingest_siu_asks_for_out_before_downloading(monkeypatch):
    import pytest

    from morie.ingest import siu

    monkeypatch.setattr(siu, "list_reports", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    with pytest.raises(SystemExit) as e:
        siu.cli(["--report-id", "17-OVI-201"])
    assert e.value.code == 2


def test_selftest_dataset_check_fetches_nothing(monkeypatch):
    from morie import datahub, hosted, selftest

    monkeypatch.setattr(hosted, "hosted_key", lambda: "sk-test")
    monkeypatch.setattr(datahub, "hosted_manifest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    assert selftest._test_datasets().startswith("Datasets:")


def test_r_module_synthetic_notes_reach_the_user(monkeypatch, tmp_path):
    import subprocess

    import pytest

    import morie.modules as m

    monkeypatch.setattr(m, "_rscript_bin", lambda: "Rscript")
    monkeypatch.setattr(m, "_R_PACKAGE", "morie")
    note = "mapq-psychometrics: runs on the deterministic synthetic MAPQII panel (n = 400); not findings"
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", note + "\n"))
    monkeypatch.setattr(m, "_load_written_outputs", lambda name, out: {})
    with pytest.warns(UserWarning, match="synthetic MAPQII panel"):
        m._run_r_module("mapq-psychometrics", cpads_csv=str(tmp_path / "x.csv"), output_dir=tmp_path)


def test_french_siu_reports_give_the_service_the_dates_and_the_person():
    from morie.siu.native import parse_report_html

    html = (
        "<html><body><p>Mandat de l'UES</p><p>Exercice du mandat</p>"
        "<p>Le pr\u00e9sent rapport porte sur la blessure grave subie par un homme de 21 ans (plaignant).</p>"
        "<p>L\u2019enqu\u00eate</p><p>Notification de l' UES [1]</p>"
        "<p>Le 29 juillet 2024, \u00e0 0 h 59, le Service de police des parcs du Niagara a communiqu\u00e9 "
        "les renseignements suivants \u00e0 l' UES .</p>"
        "<p>Le 28 juillet 2024, autour de 22 h 40, un agent du Service de police des parcs du Niagara conduisait.</p>"
        "<p>L'\u00e9quipe</p><p>Date et heure de l'envoi de l'\u00e9quipe : Le 29 juillet 2024, \u00e0 2 h 7</p>"
        "<p>T\u00e9moins civils</p><p>Agents impliqu\u00e9s</p>"
        "<p>Date : Le 26 novembre 2024</p></body></html>"
    )
    p = parse_report_html(html)
    assert p["_language"] == "fr"
    assert p["police_service"] == "Service de police des parcs du Niagara"
    assert (p["date_of_incident_iso"], p["date_siu_notified_iso"], p["date_of_director_decision_iso"]) == (
        "2024-07-28",
        "2024-07-29",
        "2024-11-26",
    )
    assert (p["age_affected"], p["sex_gender_affected"]) == ("21", "man")


def test_brent_root_converges_on_the_root_not_the_bracket():
    from morie.fn import brtmh

    want = 2 ** (1 / 3)
    for a, b in [(1.0, 2.0), (0.0, 2.0), (2.0, 1.0), (1.2, 1.3)]:
        root, info = brtmh(lambda x: x**3 - 2, a, b, full_output=True)
        assert abs(root - want) < 1e-6 and info["converged"]
        assert info["final_residual"] == abs(root**3 - 2)
    assert brtmh(lambda x: x, 0.0, 1.0) == 0.0


def test_corrcoef_and_cov_stack_the_rows_of_x_and_y_like_numpy():
    import math

    from morie.fn import _array_core as np

    x = [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]]
    y = [[4.0, 3.0, 2.0, 1.5]]
    rows = x + y

    def cov(a, b):
        ma, mb = sum(a) / 4, sum(b) / 4
        return sum((u - ma) * (v - mb) for u, v in zip(a, b)) / 3

    C = np.cov(x, y).tolist()
    R = np.corrcoef(x, y).tolist()
    assert len(C) == 3 and len(R) == 3
    for i in range(3):
        for j in range(3):
            assert abs(C[i][j] - cov(rows[i], rows[j])) < 1e-12
            r = cov(rows[i], rows[j]) / math.sqrt(cov(rows[i], rows[i]) * cov(rows[j], rows[j]))
            assert abs(R[i][j] - r) < 1e-12
    # rowvar=False: the columns are the variables (n = 3 observations, so ddof 1 divides by 2)
    Ct = np.cov([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]], [[1.0], [0.0], [2.0]], rowvar=False).tolist()
    a, c = [1.0, 3.0, 4.0], [1.0, 0.0, 2.0]
    want = sum((u - sum(a) / 3) * (v - sum(c) / 3) for u, v in zip(a, c)) / 2
    assert len(Ct) == 3 and abs(Ct[0][2] - want) < 1e-12


def test_object_array_from_nested_lists_is_two_dimensional_and_astype_str_works():
    from morie.fn import _array_core as np
    from morie.fn import geron_imputation_median

    o = np.array([["a"], ["b"], [None], ["a"]], dtype=object)
    assert o.shape == (4, 1)
    assert list(o[:, 0]) == ["a", "b", None, "a"]
    assert list(np.array(["x", 1, None], dtype=object).astype(str)) == ["x", "1", "None"]
    r = geron_imputation_median(o)
    assert [str(v) for v in r["X_imputed"].ravel()] == ["a", "b", "a", "a"]  # the mode fills the gap


def test_a_python_scalar_takes_the_scalar_branch():
    from morie.fn import geron_heaviside_step, geron_prelu, kamath_mamba_ssm

    r = geron_prelu([-2.0, 3.0], 0.25)
    assert [float(v) for v in r["a"]] == [-0.5, 3.0]
    assert float(r["grad_alpha"]) == -2.0
    # h_t = exp(delta*A) h_{t-1} + delta*B x_t with A=0, B=C=1: a running sum
    assert kamath_mamba_ssm([1.0, 2.0, 3.0], [0.0], [1.0], [1.0], 1.0)["y"] == [1.0, 3.0, 6.0]
    assert geron_heaviside_step(-1.0)["output"] == 0.0


def test_an_unbounded_dual_is_reported_not_an_overflow():
    from morie.fn import boyd_dual_problem

    r = boyd_dual_problem(lambda lam, nu: lam[0] ** 2, n_lambda=1)
    assert r["unbounded"] is True and r["dual_value"] == float("inf")
    assert r["concave"] is False
    ok = boyd_dual_problem(lambda lam, nu: -(lam[0] ** 2) + lam[0] - 1.0, n_lambda=1)
    assert ok["unbounded"] is False and abs(float(ok["lambda_"][0]) - 0.5) < 1e-6


def test_cox_fit_single_pass_matches_the_partial_likelihood_by_definition():
    import math

    from morie.fn import _array_core as np
    from morie.fn._surv import cox_fit

    t = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]
    e = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    x = [0.5, -1.0, 0.2, 1.5, -0.3, 0.8, 0.1, -0.6]

    def efron_ll(b):  # Efron's partial log-likelihood, written out from its definition
        ll = 0.0
        for u in sorted({ti for ti, ei in zip(t, e) if ei == 1}):
            risk = [math.exp(b * xi) for ti, xi in zip(t, x) if ti >= u]
            died = [i for i in range(len(t)) if t[i] == u and e[i] == 1]
            dsum = sum(math.exp(b * x[i]) for i in died)
            ll += sum(b * x[i] for i in died)
            for k in range(len(died)):
                ll -= math.log(sum(risk) - k / len(died) * dsum)
        return ll

    beta, ll, _I, _U, _n, conv = cox_fit(np.array(t), np.array(e), np.array([[v] for v in x]))
    b = float(beta[0])
    assert conv
    assert abs(ll - efron_ll(b)) < 1e-9
    h = 1e-5  # the score is zero at the maximum
    assert abs((efron_ll(b + h) - efron_ll(b - h)) / (2 * h)) < 1e-6


def test_license_metadata_matches_the_package_license():
    import pytest

    from morie import check_plugin_license, morie_license_metadata

    assert morie_license_metadata()["spdx"] == "AGPL-3.0-or-later"
    assert check_plugin_license("Apache-2.0") and check_plugin_license("GPL-3.0-or-later")
    with pytest.raises(ValueError, match="cannot be combined with morie"):
        check_plugin_license("GPL-2.0-only", raise_on_incompatible=True)


def test_a_title_row_above_the_header_is_dropped_and_stacked_tables_are_cut():
    from morie.data import _xlsx_promote_header
    from morie.fn import _frame_core as pd

    nan = float("nan")
    df = pd.DataFrame(
        {
            "Table 1 Hospital stays": ["Jurisdiction", "Canada", "Ontario", nan, "Table 1b", "Jurisdiction"],
            "Unnamed: 1": ["Number of stays", "196717", "70000", nan, nan, "Number"],
            "Unnamed: 2": ["Rate", "1.5", "2.0", nan, nan, "Rate"],
        }
    )
    out = _xlsx_promote_header(df)
    assert list(out.columns) == ["Jurisdiction", "Number of stays", "Rate"]
    assert out.shape == (2, 3)
    assert out["Number of stays"].tolist() == [196717, 70000]
    assert out["Rate"].tolist() == [1.5, 2.0]
    plain = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert _xlsx_promote_header(plain) is plain


def test_run_module_removes_only_the_staging_directory_it_made(monkeypatch, tmp_path):
    from morie import modules as m

    monkeypatch.chdir(tmp_path)
    keep = tmp_path / "keep.txt"
    keep.write_text("mine")
    # a staged path that is the caller's own relative file: its parent is ".", which must survive
    monkeypatch.setattr(m, "_cpads_csv_for_run", lambda *a, **k: "x.csv")
    monkeypatch.setattr(m, "_run_module_on", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(m, "_r_route_ready", lambda: None)
    assert m.run_module("power-design") == {"ok": True}
    assert keep.read_text() == "mine"
    # a directory the staging step made is removed after the run
    made = tmp_path / "staged-by-morie"
    made.mkdir()
    (made / "d.csv").write_text("a\n1\n")
    m._STAGED_DIRS.add(made)
    monkeypatch.setattr(m, "_cpads_csv_for_run", lambda *a, **k: made / "d.csv")
    m.run_module("power-design")
    assert not made.exists() and made not in m._STAGED_DIRS


def test_verify_and_inspect_of_a_module_that_writes_no_tables(tmp_path, capsys, monkeypatch):
    import sys

    from morie import runner

    (tmp_path / "power_summary.csv").write_text("a,b\n1,2\n")
    for verb in ("verify", "inspect"):
        monkeypatch.setattr(sys, "argv", ["morie", verb, str(tmp_path), "--module", "figures"])
        assert runner.main() == 0
        assert "figures writes no tables" in capsys.readouterr().out
        monkeypatch.setattr(sys, "argv", ["morie", verb, str(tmp_path), "--module", "descriptive-statistics"])
        assert runner.main() == 1
        assert "no table of descriptive-statistics" in capsys.readouterr().out
