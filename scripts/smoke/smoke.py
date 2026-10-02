# SPDX-License-Identifier: AGPL-3.0-or-later
"""Clean-user smoke suite: every `morie` verb run for real, in an empty HOME, with assertions.

No mocks. This is what a person gets after `pip install morie` on a machine that has
nothing on it: no datasets, no cache, no key (unless MORIE_SMOKE_KEY is set, which
unlocks the hosted-tier and data.rmorie.com cases). Run it with the package installed:

    python scripts/smoke/smoke.py            # exit 0 = every case passed
    python scripts/smoke/smoke.py --list     # the cases

Verb coverage is enforced: every subcommand of `morie --help` must have a case here
or be in NOT_RUNNABLE with a reason, otherwise the suite fails before running.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NOT_RUNNABLE: dict[str, str] = {}  # every verb has a case; a verb that cannot work here must say so honestly

KEY = os.environ.get("MORIE_SMOKE_KEY", "").strip()
RESULTS: list[tuple[str, str, str]] = []  # (verb, status, detail)


class Smoke:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.work = home / "work"
        self.work.mkdir()
        self.env = dict(os.environ)
        self.env.update(
            {
                "HOME": str(home),
                "USERPROFILE": str(home),
                "XDG_CONFIG_HOME": str(home / "cfg"),
                "XDG_CACHE_HOME": os.environ.get("MORIE_SMOKE_CACHE", str(home / "cache")),
                "MORIE_EMISSIONS_OFFLINE": "1",
                "MORIE_NO_UPDATE_CHECK": "1",
                "MORIE_HOSTED_KEY": KEY,
                "PYTHONIOENCODING": "utf-8",
            }
        )
        for var in ("LLM_API_BASE_URL", "LLM_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "MORIE_DATA_DIR"):
            self.env.pop(var, None)

    def run(self, *args: str, stdin: str = "", timeout: int = 900) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m", "morie.runner", *args]
        return subprocess.run(
            cmd,
            cwd=self.work,
            env=self.env,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

    def run_llm(self, *args: str) -> subprocess.CompletedProcess:
        """A hosted-tier call. Every runner of every package asks on the shared key at once, so a 429 is retried once after a pause."""
        r = self.run(*args)
        if r.returncode != 0 and "429" in (r.stdout + r.stderr):
            time.sleep(45)
            r = self.run(*args)
        return r


def case(verb: str):
    def deco(fn):
        fn.verb = verb
        CASES.append(fn)
        return fn

    return deco


CASES: list = []


def _lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(what)


# ------------------------------------------------------------------ cases ---


@case("list-modules")
def c_list_modules(s: Smoke):
    r = s.run("list-modules")
    check(r.returncode == 0 and "power-design" in r.stdout, r.stdout[-300:] + r.stderr[-300:])


@case("list-datasets")
def c_list_datasets(s: Smoke):
    r = s.run("list-datasets")
    check(r.returncode == 0 and "ocp21" in r.stdout, r.stderr[-300:])
    if KEY:
        check("data.rmorie.com" in r.stdout, "hosted rows missing although a key is set")


@case("cheatsheet")
def c_cheatsheet(s: Smoke):
    r = s.run("cheatsheet")
    check(r.returncode == 0 and "morie login" in r.stdout and "provider set" in r.stdout, r.stdout[:200])


@case("explain")
def c_explain(s: Smoke):
    r = s.run("explain", "power_two_proportion_gender.csv")
    check(r.returncode == 0 and "effect_size" in r.stdout, r.stdout[:200])


@case("doctor")
def c_doctor(s: Smoke):
    r = s.run("doctor")
    check(r.returncode in (0, 1) and "llm" in r.stdout.lower(), r.stdout[:300] + r.stderr[:300])


@case("models")
def c_models(s: Smoke):
    r = s.run("models")
    check(r.returncode == 0, r.stderr[-300:])
    if not KEY:
        check("login" in (r.stdout + r.stderr).lower(), "without a key, models must point at morie login")
    else:
        # the gateway serves Cloudflare Workers AI models beside the ollama.com ones
        check(":cf" in r.stdout, "no Workers AI model listed: " + r.stdout[-300:])


@case("provider")
def c_provider(s: Smoke):
    r = s.run(
        "provider", "set", "--base-url", "https://api.example.org/v1/", "--key", "sk-smoke-1234567", "--model", "demo"
    )
    check(r.returncode == 0, r.stderr[-300:])
    r = s.run("provider", "show")
    check("api.example.org/v1" in r.stdout and "sk-smoke-1234567" not in r.stdout, r.stdout)
    r = s.run("models")
    check("Your endpoint" in r.stdout, r.stdout[:300])
    r = s.run("provider", "unset")
    check(r.returncode == 0, r.stderr[-300:])
    check("No endpoint" in s.run("provider", "show").stdout, "unset did not clear the endpoint")


def _answered(r, verb: str) -> None:
    """With a key: a model answered (exit 0). Without: the local fallback text, exit 1 by design. Never empty."""
    check(r.stdout.strip(), f"{verb} printed nothing: " + r.stderr[-300:])
    if KEY:
        check(
            r.returncode == 0,
            f"{verb} with a key did not get a model answer: stdout={r.stdout[-200:]!r} stderr={r.stderr[-300:]!r}",
        )
    else:
        check(
            r.returncode == 1 and "fallback" in r.stderr,
            f"{verb} without a key must say it fell back: " + r.stderr[-300:],
        )


@case("ask")
def c_ask(s: Smoke):
    _answered(s.run_llm("ask", "--no-stream", "What does the power-design module compute?"), "ask")
    # a Cloudflare Workers AI model, named per call
    r = s.run_llm("ask", "--no-stream", "--model", "gpt-oss-120b:cf", "Reply with the single word pong.")
    if KEY:
        _answered(r, "ask --model gpt-oss-120b:cf")
    else:
        check(r.returncode != 0, "without a key, ask must not claim an answer")


def _agent_case(s: Smoke, verb: str, *args: str) -> None:
    """The agent layer (agent.py) is kept out of the published wheel on purpose; from a wheel the verb must say so and exit 1."""
    r = s.run_llm(verb, "--no-stream", *args)
    if _source_tree_only(r.stdout + r.stderr):
        check(r.returncode == 1, f"{verb} without the agent layer must exit 1: " + r.stdout[-200:])
        RESULTS.append(
            (verb, "SKIP", "the agent layer is excluded from the wheel by design; the honest message was verified")
        )
        return
    _answered(r, verb)


@case("percy")
def c_percy(s: Smoke):
    _agent_case(s, "percy", "Which module compares two groups?")


@case("perseus")
def c_perseus(s: Smoke):
    _agent_case(s, "perseus", "hello")


@case("agent")
def c_agent(s: Smoke):
    _answered(s.run_llm("agent", "--no-stream", "hello"), "agent")


def _pty_session(s: Smoke, args: list[str], lines: list[str], timeout: int = 120) -> str:
    """Run a verb that insists on a terminal under a pseudo-terminal (POSIX) and type into it."""
    import pty
    import select
    import signal

    pid, fd = pty.fork()
    if pid == 0:  # child
        os.chdir(s.work)
        os.environ.update(s.env)
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = "120"
        os.environ["LINES"] = "40"
        os.execv(sys.executable, [sys.executable, "-m", "morie.runner", *args])
    out = b""
    deadline = time.time() + timeout
    sent = 0
    try:
        while time.time() < deadline:
            rl, _, _ = select.select([fd], [], [], 1.0)
            if rl:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
            if sent < len(lines) and (time.time() > deadline - timeout + 3 * (sent + 1)):
                os.write(fd, (lines[sent] + "\r").encode())
                sent += 1
            if sent == len(lines) and not rl and time.time() > deadline - timeout + 3 * (sent + 2):
                break
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        with contextlib.suppress(OSError):
            os.waitpid(pid, 0)
    return out.decode("utf-8", "replace")


@case("chat")
def c_chat(s: Smoke):
    if os.name == "nt":
        RESULTS.append(("chat", "SKIP", "pseudo-terminal cases run on POSIX"))
        return
    text = _pty_session(s, ["chat"], ["/help", "/quit"])
    check(
        "/quit" in text or "bye" in text.lower() or "chat" in text.lower(),
        "chat did not start under a pty: " + text[-300:],
    )
    check("Traceback" not in text, "chat crashed: " + text[-400:])


@case("tui")
def c_tui(s: Smoke):
    if os.name == "nt":
        RESULTS.append(("tui", "SKIP", "pseudo-terminal cases run on POSIX"))
        return
    text = _pty_session(s, ["tui"], ["q"], timeout=60)
    check("Traceback" not in text, "tui crashed: " + text[-400:])
    if "interactive" in text.lower() and "pip install" in text.lower():
        RESULTS.append(("tui", "SKIP", "the interactive extra is not installed"))


@case("repl")
def c_repl(s: Smoke):
    script = "x = 41\nprint(x + 1)\n"
    r = s.run("repl", stdin=script, timeout=300)
    if _source_tree_only(r.stdout + r.stderr):
        check(r.returncode != 0, "repl must fail when it is not in this build")
        RESULTS.append(("repl", "SKIP", "the polyglot REPL is source-tree only"))
        return
    check(r.returncode == 0 and "42" in r.stdout, "python line in the REPL: " + r.stdout[-300:] + r.stderr[-300:])
    if shutil.which("Rscript"):
        # polyglot: a Python value bridged into R in the same session
        r = s.run("repl", stdin="x = 20\nR> cat(x * 2 + 2)\n", timeout=300)
        check(
            r.returncode == 0 and "42" in r.stdout, "polyglot bridge Python -> R: " + r.stdout[-400:] + r.stderr[-300:]
        )
    else:
        RESULTS.append(("repl polyglot", "SKIP", "no Rscript on this runner"))


@case("tutorial")
def c_tutorial(s: Smoke):
    r = s.run("tutorial", stdin="q\n", timeout=120)
    check(r.returncode == 0 and "tutorial" in r.stdout.lower(), r.stdout[:200] + r.stderr[-200:])


@case("login")
def c_login(s: Smoke):
    if not KEY:
        RESULTS.append(("login", "SKIP", "MORIE_SMOKE_KEY not set"))
        return
    r = s.run("login", "--token", KEY)
    check(r.returncode == 0 and "accepted" in (r.stdout + r.stderr).lower() or r.returncode == 0, r.stdout + r.stderr)
    check((s.home / "cfg" / "morie" / "credentials.json").exists(), "credentials file not written")


@case("interactive")
def c_interactive(s: Smoke):
    """The source-tree-only layer can be added to an installed copy: status, offline install from this checkout, exec works, remove."""
    r = s.run("interactive", "status")
    check(r.returncode == 0 and "interactive layer directory" in r.stdout, r.stdout[-300:] + r.stderr[-300:])
    src = Path(__file__).resolve().parents[2] / "src" / "morie"
    r = s.run("interactive", "install", "--from", str(src))
    check(r.returncode == 0 and "Installed the interactive layer" in r.stdout, r.stdout[-400:] + r.stderr[-300:])
    r = s.run("interactive", "status")
    check("active" in r.stdout, "status after install: " + r.stdout[-300:])
    r = s.run("exec", "print(6 * 7)")
    check(r.returncode == 0 and "42" in r.stdout, "exec after install: " + r.stdout[-300:] + r.stderr[-300:])
    r = s.run("interactive", "remove")
    check(r.returncode == 0 and "Removed" in r.stdout, r.stdout[-300:] + r.stderr[-300:])
    r = s.run("exec", "print(6 * 7)")
    check(
        _source_tree_only(r.stdout + r.stderr) or "42" in r.stdout,
        "exec after remove: " + r.stdout[-300:] + r.stderr[-300:],
    )


@case("logout")
def c_logout(s: Smoke):
    r = s.run("logout")
    check(r.returncode == 0, r.stderr[-300:])


@case("generate-template")
def c_template(s: Smoke):
    r = s.run("generate-template", "--module", "hawkes", "--out", "paper/first.md")
    check(
        r.returncode == 0 and (s.work / "paper" / "first.md").read_text(encoding="utf-8").count("hawkes") > 0,
        r.stderr[-300:],
    )


def _csv(s: Smoke) -> Path:
    p = s.work / "d.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["treated", "y", "g", "w"])
        for i in range(120):
            w.writerow([i % 2, (i * 7) % 13 / 3.0, "a" if i < 60 else "b", 1 + i % 3])
    return p


@case("profile-dataset")
def c_profile(s: Smoke):
    p = _csv(s)
    r = s.run("profile-dataset", str(p), "--treatment", "treated", "--outcome", "y", "--suggest")
    check(r.returncode == 0 and "Suggested" in r.stdout, r.stdout[-300:] + r.stderr[-300:])


@case("sample")
def c_sample(s: Smoke):
    p = _csv(s)
    r = s.run("sample", str(p), "--n", "7", "--output", "s.csv")
    check(r.returncode == 0 and _lines(s.work / "s.csv") == 8, r.stdout[-200:] + r.stderr[-200:])
    r = s.run("sample", str(p), "--n", "3", "--method", "stratified", "--strata-col", "g", "--output", "st.csv")
    check(r.returncode == 0 and _lines(s.work / "st.csv") == 7, r.stdout[-200:] + r.stderr[-200:])


@case("run-module")
def c_run_module_before(s: Smoke):
    r = s.run("run-module", "power-design", "--output-dir", "out0")
    check(r.returncode == 0 and (s.work / "out0").exists(), r.stdout[-300:] + r.stderr[-400:])
    s.before = (s.work / "out0" / "power_two_proportion_gender.csv").exists()
    check(s.before, "power-design wrote no power_two_proportion_gender.csv")


@case("pull")
def c_pull(s: Smoke):
    r = s.run("pull", "ocp21", "--out", "cpads.csv", timeout=1800)
    check(r.returncode == 0, "pull ocp21 failed: " + r.stderr[-400:])
    rows = _lines(s.work / "cpads.csv") - 1
    check(rows > 40000, f"CPADS PUMF has {rows} rows; expected ~40,931")
    if KEY:
        r = s.run("pull", "fec_cm_2020/fec_cm_2020", "--out", "fec.csv", timeout=900)
        check(
            r.returncode == 0 and _lines(s.work / "fec.csv") > 1000,
            "data.rmorie.com pull failed: " + r.stderr[-400:] + r.stdout[-200:],
        )
    else:
        RESULTS.append(("pull data.rmorie.com", "SKIP", "MORIE_SMOKE_KEY not set"))
    if not _r_arm_ok():
        RESULTS.append(
            ("pull -> module", "SKIP", "no Rscript, or no R 'morie' package, on this runner (the modules are R-backed)")
        )
        return
    r = s.run("run-module", "descriptive-statistics", "--output-dir", "out1")
    check(r.returncode == 0, r.stderr[-400:])
    check("SYNTHETIC" not in r.stderr.upper(), "after the pull the module still used the synthetic frame")


@case("run-modules")
def c_run_modules(s: Smoke):
    r = s.run("run-modules", "--modules", "power-design", "--output-dir", "out2")
    check(r.returncode == 0 and "power-design" in r.stdout, r.stderr[-300:])


@case("pipeline")
def c_pipeline(s: Smoke):
    r = s.run("pipeline", "--modules", "power-design", "-y", "--output-dir", "out3")
    check(r.returncode == 0, r.stderr[-400:])
    check("emissions" in r.stdout.lower(), "pipeline printed no emissions line")
    check((s.work / "out3" / "emissions" / "emissions.csv").exists(), "no emissions.csv under the output dir")


@case("inspect")
def c_inspect(s: Smoke):
    r = s.run("inspect", "out0")
    check(r.returncode == 0 and "rows" in r.stdout.lower(), r.stdout[:200] + r.stderr[-200:])


@case("verify")
def c_verify(s: Smoke):
    r = s.run("verify", "out0")
    check(r.returncode in (0, 1) and r.stdout.strip(), r.stderr[-300:])


@case("emissions")
def c_emissions(s: Smoke):
    r = s.run("emissions", "--seconds", "1", "--output-dir", "em", "--country", "CAN")
    check(r.returncode == 0 and "kg CO2eq" in r.stdout and "Capsule" in r.stdout, r.stdout + r.stderr[-300:])
    check(
        (s.work / "em" / "emissions.csv").exists() and (s.work / "em" / "emissions_manifest.json").exists(),
        "emissions files missing",
    )
    m = json.loads((s.work / "em" / "emissions_manifest.json").read_text(encoding="utf-8"))
    check(m["meta"]["measurements"]["emissions"] > 0, "emissions not positive")


@case("verify-pollution")
def c_pollution(s: Smoke):
    r = s.run("verify-pollution", "--pollutant", "no2", "--demo")
    check(r.returncode == 0 and "STATUS: ok" in r.stdout and "source:   Atkinson" in r.stdout, r.stdout[-400:])
    r = s.run("verify-pollution", "--pollutant", "pm25", "--exposure-mean", "2", "--exposure-prevalence", "0.5")
    check(r.returncode == 1 and "assumption_failure" in r.stdout, "assumption failure not reported")
    r = s.run("verify-pollution", "--pollutant", "no2", "--demo", "--json")
    check(json.loads(r.stdout)["pipeline"]["paf"] > 0, "json mode")


@case("crypto")
def c_crypto(s: Smoke):
    r = s.run("crypto", "keygen", "--name", "smoke", "--output", "keys")
    if r.returncode != 0 and "liboqs" in (r.stdout + r.stderr):
        RESULTS.append(("crypto", "SKIP", "built without liboqs"))
        return
    check(r.returncode == 0 and (s.work / "keys" / "smoke.moriepk").exists(), r.stdout + r.stderr[-300:])
    (s.work / "secret.txt").write_text("hello", encoding="utf-8")
    r = s.run("crypto", "encrypt", "secret.txt", "--to", "keys/smoke.moriepk")
    check(r.returncode == 0 and (s.work / "secret.txt.morieenc").exists(), r.stdout + r.stderr[-300:])


@case("ingest")
def c_ingest(s: Smoke):
    r = s.run("ingest", "tps", "--list")
    check(r.returncode == 0 and r.stdout.strip(), r.stderr[-300:])


@case("download-bootstrap")
def c_bootstrap(s: Smoke):
    r = s.run("download-bootstrap", "--survey", "zzz_1999")
    check(r.returncode == 2 and "invalid choice" in r.stderr, "unknown survey must be rejected: " + r.stderr[-300:])
    r = s.run("download-bootstrap", "--survey", "csus_2023", "--limit", "50", timeout=1800)
    check(r.returncode == 0 and "OK" in r.stdout, r.stdout[-300:] + r.stderr[-300:])


@case("percysuits")
def c_percysuits(s: Smoke):
    r = s.run("percysuits", "--dry-run")
    check(r.returncode in (0, 1) and r.stdout.strip(), r.stderr[-300:])


_R_ARM: bool | None = None


def _r_arm_ok() -> bool:
    """Rscript on PATH and the R 'morie' package installed: what the R-backed modules need."""
    global _R_ARM
    if _R_ARM is None:
        _R_ARM = False
        if shutil.which("Rscript"):
            try:
                r = subprocess.run(
                    ["Rscript", "-e", "quit(status = as.integer(!requireNamespace('morie', quietly = TRUE)))"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                _R_ARM = r.returncode == 0
            except (OSError, subprocess.SubprocessError):
                _R_ARM = False
    return _R_ARM


def _source_tree_only(text: str) -> bool:
    """The exec/REPL layer is excluded from the wheel on purpose; its verbs must refuse honestly."""
    return "not bundled in this install" in text or "not available in this build" in text


@case("edit")
def c_edit(s: Smoke):
    stub = s.work / "stub_editor.py"
    stub.write_text("import sys\nopen(sys.argv[1], 'a').write('print(6 * 7)\\n')\n")
    env = dict(s.env, EDITOR=f'"{sys.executable}" "{stub}"')
    env.pop("VISUAL", None)
    r = subprocess.run(
        [sys.executable, "-m", "morie.runner", "edit", "f.py", "--run"],
        cwd=s.work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    check("print(6 * 7)" in (s.work / "f.py").read_text(), "the editor stub did not write the file: " + r.stderr[-300:])
    if _source_tree_only(r.stdout + r.stderr):
        check(r.returncode != 0, "--run must fail when exec is not in this build")
        RESULTS.append(("edit --run", "SKIP", "exec is source-tree only"))
    else:
        check(r.returncode == 0 and "42" in r.stdout, "edit --run: " + r.stdout[-200:] + r.stderr[-300:])


@case("serve")
def c_serve(s: Smoke):
    import socket
    import urllib.error
    import urllib.request

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "morie.runner", "serve", "--port", str(port)],
        cwd=s.work,
        env=s.env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health = None
        for _ in range(60):
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(base + "/v1/health", timeout=5) as h:
                    health = json.load(h)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        check(proc.poll() is None and bool(health) and health.get("status") == "ok", f"relay did not come up on {port}")
        req = urllib.request.Request(
            base + "/v1/percy",
            data=json.dumps({"question": "Reply with the single word pong."}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                code, body = resp.status, json.load(resp)
        except urllib.error.HTTPError as exc:
            code, body = exc.code, json.load(exc)
        if KEY:
            check(
                code == 200 and body.get("text", "").strip() and "LLM request failed" not in body["text"],
                f"relay answer with a key: {code} {str(body)[:300]}",
            )
        else:
            check(
                code == 503 or (code == 200 and body.get("backend") not in (None, "local_fallback")),
                f"without a key the relay must say 503, not pretend: {code} {str(body)[:300]}",
            )
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


@case("verify-earth-engine")
def c_verify_earth_engine(s: Smoke):
    """Without the Earth Engine client or credentials the verb reports the failed stage and exits non-zero."""
    r = s.run("verify-earth-engine", timeout=300)
    out = r.stdout + r.stderr
    check("Traceback" not in out, "verify-earth-engine crashed: " + out[-400:])
    check(
        r.returncode == 0
        or ("STATUS" in out and ("earthengine" in out or "credential" in out.lower() or "FAIL" in out)),
        out[-400:],
    )


@case("bricklayer")
def c_bricklayer(s: Smoke):
    """Status mode: reports what is installed and never installs without --yes."""
    r = s.run("bricklayer")
    check(r.returncode == 0 and "morie family status" in r.stdout, r.stdout[-300:] + r.stderr[-300:])


@case("r-install")
def c_r_install(s: Smoke):
    r = s.run("r-install")
    check(r.returncode == 0 and "morie family status" in r.stdout, r.stdout[-300:] + r.stderr[-300:])


@case("parity-review")
def c_parity_review(s: Smoke):
    """Maintainer tool: with no reference checkout it says so instead of a traceback."""
    empty = s.work / "empty-tree"
    empty.mkdir(exist_ok=True)
    r = s.run("parity-review", "--epiml-root", str(empty))
    check(r.returncode == 2 and "Traceback" not in r.stderr and "parity-review:" in r.stderr, r.stderr[-300:])
    r = s.run("parity-review", "--epiml-root", str(s.work / "does-not-exist"))
    check(r.returncode == 2 and "not a directory" in r.stderr, r.stderr[-300:])


@case("convert-checkpoint")
def c_convert_checkpoint(s: Smoke):
    """A missing checkpoint and the trust gate are reported in words."""
    r = s.run("convert-checkpoint", "--checkpoint", "missing.pt", "--output", "out.gguf")
    check(r.returncode == 2 and "checkpoint not found" in r.stderr, r.stderr[-300:])
    (s.work / "fake.pt").write_bytes(b"not a checkpoint")
    r = s.run("convert-checkpoint", "--checkpoint", "fake.pt", "--output", "out.gguf")
    check(r.returncode != 0 and "Traceback" not in r.stderr and "convert-checkpoint:" in r.stderr, r.stderr[-300:])


@case("exec")
def c_exec(s: Smoke):
    r = s.run("exec", "--lang", "python", "print(6*7)")
    if _source_tree_only(r.stdout + r.stderr):
        check(r.returncode != 0, "exec must fail when it is not in this build")
        RESULTS.append(("exec", "SKIP", "exec is source-tree only"))
        return
    check(r.returncode == 0 and "42" in r.stdout, r.stdout + r.stderr[-200:])
    if shutil.which("Rscript"):
        r = s.run("exec", "--lang", "r", "cat(6 * 7)")
        check(r.returncode == 0 and "42" in r.stdout, "exec --lang r: " + r.stdout + r.stderr[-300:])
    else:
        RESULTS.append(("exec r", "SKIP", "no Rscript on this runner"))


@case("selftest")
def c_selftest(s: Smoke):
    r = s.run("selftest", timeout=900)
    if not _r_arm_ok():
        # the R-backed checks cannot pass without Rscript and the R package; every other row must
        rows = [ln for ln in r.stdout.splitlines() if ln.strip().startswith(("FAIL", "OK", "SKIP"))]
        bad = [ln for ln in rows if ln.strip().startswith("FAIL") and "R " not in ln and "module" not in ln.lower()]
        check(not bad, "selftest failed a check that does not need R: " + "\n".join(bad)[-400:])
        return
    check(r.returncode == 0 and "FAILED" not in r.stdout, r.stdout[-600:])


@case("readme-python")
def c_readme_python(s: Smoke):
    """Every ```python block in README.md runs as written (the install and shell blocks are not Python)."""
    import re

    readme = Path(__file__).resolve().parents[2] / "README.md"
    if not readme.exists():
        RESULTS.append(("readme-python", "SKIP", "README.md not next to the suite (installed copy)"))
        return
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(encoding="utf-8"), re.S)
    check(blocks, "README has no python blocks")
    for i, code in enumerate(blocks, 1):
        r = subprocess.run(
            [sys.executable, "-c", code], cwd=s.work, env=s.env, capture_output=True, text=True, timeout=900
        )
        check(r.returncode == 0, f"README python block {i} failed: " + r.stderr[-500:])


@case("update")
def c_update(s: Smoke):
    r = s.run("update")
    check(r.returncode == 0, r.stdout[-200:] + r.stderr[-200:])


# ------------------------------------------------------------- the runner ---


def verbs_from_parser() -> set[str]:
    from morie.runner import build_parser

    for a in build_parser()._actions:
        if isinstance(a, argparse._SubParsersAction):
            return set(a.choices)
    return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    if a.list:
        for c in CASES:
            print(c.verb)
        return 0
    verbs = verbs_from_parser()
    extra = {"readme-python"}  # cases that are not CLI verbs
    covered = {c.verb for c in CASES} | set(NOT_RUNNABLE)
    missing = sorted(verbs - covered)
    if missing:
        print("VERBS WITHOUT A SMOKE CASE:", ", ".join(missing))
        return 2
    stale = sorted(covered - verbs - extra)
    if stale:
        print("CASES FOR VERBS THAT NO LONGER EXIST:", ", ".join(stale))
        return 2
    home = Path(tempfile.mkdtemp(prefix="morie-smoke-"))
    s = Smoke(home)
    failed = 0
    for c in CASES:
        if a.only and c.verb not in a.only:
            continue
        t0 = time.time()
        try:
            c(s)
            RESULTS.append((c.verb, "OK", f"{time.time() - t0:.1f}s"))
        except Exception as exc:  # noqa: BLE001 - every failure is reported, none hides the rest
            failed += 1
            RESULTS.append((c.verb, "FAIL", f"{type(exc).__name__}: {str(exc)[:600]}"))
    for verb, st, detail in RESULTS:
        print(f"[{st:4}] {verb:<22} {detail}")
    print(
        f"\nsmoke: {sum(1 for r in RESULTS if r[1] == 'OK')} ok, {failed} failed, "
        f"{sum(1 for r in RESULTS if r[1] == 'SKIP')} skipped, {len(NOT_RUNNABLE)} not runnable here"
    )
    shutil.rmtree(home, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
