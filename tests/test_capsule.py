# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.capsule: provenance, manifests, HMAC bundles and drift, and the
cross-checks against rmoriebricklayer (run when R and the package are present)."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import random
import shutil
import subprocess

import pytest

from morie import capsule as cps
from morie.capsule import _json
from morie.capsule._prov import safe_rel

KEY = "a shared secret for the tests"

# ------------------------------------------------------------------ helpers


def _csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def _capsule(d, n=5):
    d.mkdir(exist_ok=True)
    _csv(d / "d.csv", ["id", "grp", "x"], [[i, "ab"[i % 2], i / 3] for i in range(n)])
    (d / "analysis.py").write_text("print('analysis')\n")
    cps.write(d, "d.csv", "https://example.org/d.csv", retrieved="2026-10-20", script_file="analysis.py")
    return d


def _rlibs():
    return os.environ.get("RMBL_R_LIBS") or os.path.expanduser("~/tmp/Rlibside")


def _have_r() -> bool:
    if shutil.which("Rscript") is None:
        return False
    env = {**os.environ, "R_LIBS": _rlibs()}
    r = subprocess.run(
        ["Rscript", "-e", "quit(status = !requireNamespace('rmoriebricklayer', quietly = TRUE))"],
        env=env,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    return r.returncode == 0


needs_r = pytest.mark.skipif(not _have_r(), reason="Rscript with rmoriebricklayer is not available")


def _r(tmp_path, code: str):
    """Run R code that leaves its answer in `out` (a list); returns it parsed."""
    script = tmp_path / "check.R"
    res = tmp_path / "out.json"
    script.write_text(
        "suppressMessages(library(rmoriebricklayer))\n"
        + code
        + f"\nwriteLines(tojs(out, auto_unbox = TRUE, digits = I(17), na = 'null', null = 'null'), '{res}')\n"
    )
    env = {**os.environ, "R_LIBS": _rlibs()}
    p = subprocess.run(["Rscript", str(script)], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert p.returncode == 0, p.stderr
    return json.loads(res.read_text())


def _rows(chk):
    return [(c["check"], c["ok"]) for c in chk["checks"]]


# ------------------------------------------------------------------ JSON codec


def test_canon_numbers_and_strings():
    assert _json.canon({"b": 1, "a": [0.1, 3.0, 1e20, -0.0, None, True, 2**40]}) == (
        '{"a":[0.10000000000000001,3,1e+20,-0,null,true,1099511627776],"b":1}'
    )
    assert _json.canon({"x": 1 / 3}) == '{"x":0.33333333333333331}'
    assert _json.canon({"x": float("nan")}) == '{"x":null}'
    assert _json.canon('a/b"\\\n\x01é') == '"a/b\\"\\\\\\n\\u0001é"'
    assert _json.pretty({"v": [1, 2.5], "e": [], "o": {}, "l": [{"a": 1}]}) == (
        '{\n  "v": [1, 2.5],\n  "e": [],\n  "o": {},\n  "l": [\n    {\n      "a": 1\n    }\n  ]\n}'
    )


def test_safe_rel(tmp_path):
    assert safe_rel("a.csv", str(tmp_path)) == f"{tmp_path}/a.csv"
    for bad in ("../x", "/etc/passwd", "~/x", "C:/x", "a//b", "./a", "", None, ["../x"]):
        assert safe_rel(bad, str(tmp_path)) is None
    try:
        os.symlink(tmp_path.parent, tmp_path / "out", target_is_directory=True)
    except (OSError, NotImplementedError):  # Windows without the symlink privilege
        return
    assert safe_rel("out/x", str(tmp_path)) is None


# ------------------------------------------------------------------ provenance + verify


def test_write_and_verify(tmp_path):
    d = _capsule(tmp_path / "c")
    prov = cps.load(d / "data_provenance.json")
    assert list(prov) == ["name", "source", "resource", "script"]
    assert prov["resource"]["row_count_data_rows"] == 5
    chk = cps.verify(d)
    assert chk["ok"], chk
    assert [c["check"] for c in chk["checks"]] == [
        "provenance_readable",
        "data_present",
        "data_sha256",
        "data_size_bytes",
        "data_readable",
        "data_row_count",
        "script_sha256",
        "data_not_synthetic",
    ]
    assert chk["missing"] == []
    # edited data: digest, size and row count fail
    with open(d / "d.csv", "a") as fh:
        fh.write("9,a,3\n")
    bad = dict(_rows(cps.verify(d)))
    assert not bad["data_sha256"] and not bad["data_size_bytes"] and not bad["data_row_count"]
    # re-pinned, it verifies again
    cps.pin(d)
    assert cps.verify(d)["ok"]


def test_verify_failures(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "data_provenance.json").write_text("{}")
    chk = cps.verify(d)
    assert not chk["ok"] and ("data_present", False) in _rows(chk) and "data_sha256" in chk["missing"]
    assert not cps.verify(d, provenance_file="../x.json")["ok"]
    with pytest.raises(NotADirectoryError):
        cps.verify(tmp_path / "nope")
    d2 = _capsule(tmp_path / "c2")
    (d2 / "d.csv.synthetic").write_text("")
    assert ("data_not_synthetic", False) in _rows(cps.verify(d2))
    with pytest.raises(ValueError):
        cps.load(d / "data_provenance.json")


def test_schema_checks(tmp_path):
    d = _capsule(tmp_path / "c")
    p = json.loads((d / "data_provenance.json").read_text())
    p["schema"] = {"expected_columns": ["id", "nope"], "expected_types": {"x": "character"}}
    (d / "data_provenance.json").write_text(json.dumps(p))
    rows = {c["check"]: c for c in cps.verify(d)["checks"]}
    assert not rows["schema_valid"]["ok"]
    assert "Missing required columns: nope" in rows["schema_valid"]["detail"]
    assert "Column 'x' is numeric, expected character" in rows["schema_valid"]["detail"]


# ------------------------------------------------------------------ manifests


def test_manifest_record_compare(tmp_path):
    m = cps.manifest({"project": "p"}, env=False)
    assert m["results"] == [] and cps.canon(cps.manifest(env=False)) == '{"meta":[],"results":[]}'
    cps.record(m, "a", 1 / 3, 0.3333, tol=0.001)
    cps.record(m, "b", 2.5, 2.0, tol=0.01)
    cps.record(m, "c", "CA", "CA")
    assert [r["status"] for r in m["results"].values()] == ["PASS", "DIFFER", "INFO"]
    p = cps.write_manifest(m, tmp_path / "m.json")
    assert cps.read_manifest(p) == m
    c = cps.write_manifest(m, tmp_path / "c.json", canonical=True)
    assert pathlib.Path(c).read_text() == cps.canon(m) + "\n"
    res = cps.compare(m, None, {"a": lambda _: 1 / 3, "c": lambda _: "CA"})
    assert [r["status"] for r in res["results"]] == ["MATCH", "MATCH"] and res["unchecked"] == ["b"]
    assert not res["ok"]
    res = cps.compare(m, None, {"a": lambda _: 1 / 0, "zz": lambda _: 1})
    assert [r["status"] for r in res["results"]] == ["ERROR", "NOT_RECORDED"]


def test_manifest_synthetic(monkeypatch):
    monkeypatch.setenv("BRICKLAYER_SYNTHETIC", "1")
    m = cps.record(cps.manifest(env=False), "a", 1, 1)
    assert m["meta"]["synthetic"] is True and m["results"]["a"]["status"] == "INFO"


# ------------------------------------------------------------------ signing


def test_hmac_sign_verify():
    # RFC 4231 test case 2
    s = cps.sign("what do ya want for nothing?", "Jefe")
    assert s["signature"] == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    assert cps.verify_signature("what do ya want for nothing?", s, "Jefe")
    assert not cps.verify_signature("tampered", s, "Jefe")
    assert not cps.verify_signature("what do ya want for nothing?", s, "other")
    assert not cps.verify_signature("x", {"scheme": "hmac", "signature": "00"}, "Jefe")


def test_bundle_roundtrip(tmp_path):
    d = _capsule(tmp_path / "c")
    m = cps.record(cps.manifest({"project": "p"}, env=False), "n", 5, 5)
    cps.write_manifest(m, d / "manifest.json")
    b = cps.bundle(d, m, KEY, note="release")
    assert list(b) == ["files", "manifest_digest", "bundled_utc", "attestation"]
    assert list(b["attestation"]) == [
        "digest",
        "scheme",
        "public",
        "context",
        "prehash",
        "note",
        "signed_utc",
        "signature",
    ]
    chk = cps.verify_bundle(str(d / "capsule_bundle.json"), d, manifest=m, key=KEY, public_key=cps.key_id(KEY))
    assert chk["ok"], chk
    assert not cps.verify_bundle(str(d / "capsule_bundle.json"), d, key="wrong")["ok"]
    (d / "extra.txt").write_text("x")
    rows = dict(_rows(cps.verify_bundle(str(d / "capsule_bundle.json"), d, key=KEY)))
    assert rows["no_unlisted_files"] is False and rows["attestation:signature"] is True
    os.remove(d / "extra.txt")
    with open(d / "d.csv", "a") as fh:
        fh.write("1,a,1\n")
    assert dict(_rows(cps.verify_bundle(str(d / "capsule_bundle.json"), d, key=KEY)))["file:d.csv"] is False


# ------------------------------------------------------------------ drift


def test_drift_numeric_and_categorical(tmp_path):
    rng = random.Random(7)
    ref = _csv(tmp_path / "r.csv", ["x", "g"], [[rng.gauss(0, 1), rng.choice("abc")] for _ in range(400)])
    same = _csv(tmp_path / "s.csv", ["x", "g"], [[rng.gauss(0, 1), rng.choice("abc")] for _ in range(400)])
    moved = _csv(tmp_path / "m.csv", ["x", "h"], [[rng.gauss(2, 1), rng.choice("abc")] for _ in range(400)])
    r = cps.drift(ref, same)
    assert not r["anydrft"] and [c["type"] for c in r["columns"]] == ["numeric", "categorical"]
    r = cps.drift(ref, moved)
    assert r["anydrft"] and r["columns"][0]["drifted"] is True and r["added"] == ["h"] and r["removed"] == ["g"]
    b = cps.benford([1, 2, 3, 10, 200, 3000, 1e-310])
    assert b["counts"]["1"] == 3 and b["n"] == 7 and b["df"] == 8


def test_parity_arguments(tmp_path):
    d = _capsule(tmp_path / "c")
    p = cps.write(d, "d.csv", retrieved="2026-10-20")
    assert p["source"] == {"retrieved": "2026-10-20"} and cps.verify(d)["ok"]
    with pytest.raises(FileNotFoundError):
        cps.write(d, "../d.csv")
    m = cps.manifest(
        {"project": "p"},
        checks=[
            {"name": "n", "observed": 5, "expected": 5},
            {"name": "m", "observed": 1.0, "expected": 2.0, "tol": 0.5},
        ],
        path=tmp_path / "m.json",
        env=False,
    )
    assert [r["status"] for r in m["results"].values()] == ["PASS", "DIFFER"]
    assert cps.compare(tmp_path / "m.json", None, {"n": lambda _: 5, "m": lambda _: 1.0})["ok"]
    cur = _csv(tmp_path / "cur.csv", ["id", "grp", "x"], [[i, "ab"[i % 2], i / 3] for i in range(5)])
    assert not cps.drift(d, cur)["anydrft"]
    with pytest.raises(ValueError):
        cps.sign("m", KEY, scheme="ML-DSA-44")


# ------------------------------------------------------------------ CLI


def test_cli(tmp_path, capsys, monkeypatch):
    from morie.capsule.cli import run
    from morie.runner import build_parser

    def cli(*a):
        return run(build_parser().parse_args(["capsule", *a]))

    d = tmp_path / "c"
    d.mkdir()
    f = _csv(d / "d.csv", ["id"], [[1], [2]])
    assert cli("init", f, "--url", "https://example.org", "--json") == 0
    assert json.loads(capsys.readouterr().out)["resource"]["row_count_data_rows"] == 2
    assert cli("verify", str(d), "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    monkeypatch.setenv("MORIE_CAPSULE_KEY", KEY)
    assert cli("sign", str(d)) == 0
    assert cli("verify", str(d)) == 0
    monkeypatch.setenv("MORIE_CAPSULE_KEY", "wrong")
    assert cli("verify", str(d)) == 1
    monkeypatch.delenv("MORIE_CAPSULE_KEY")
    assert cli("sign", str(d)) == 2
    assert cli("verify", str(tmp_path / "nope")) == 2
    g = _csv(tmp_path / "g.csv", ["id"], [[5], [6]])
    assert cli("drift", f, f) == 0
    assert cli("drift", f, g, "--json") in (0, 1)


# ------------------------------------------------------------------ cross-checks with R


@needs_r
def test_r_verifies_morie_capsule(tmp_path):
    d = _capsule(tmp_path / "c")
    m = cps.record(cps.manifest({"project": "p"}, env=False), "n", 5, 5)
    cps.write_manifest(m, d / "manifest.json")
    out = _r(tmp_path, f"v <- vrfcps('{d}'); out <- list(ok = v$ok, check = v$checks$check, pass = v$checks$ok)")
    py = cps.verify(d)
    assert out["ok"] is True and py["ok"] is True
    assert out["check"] == [c["check"] for c in py["checks"]]


@needs_r
def test_morie_verifies_r_capsule(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    out = _r(
        tmp_path,
        f"""
d <- '{d}'
write.csv(data.frame(id = 1:6, g = c('a','b','a','b','a','c'), x = (1:6) / 7), file.path(d, 'd.csv'),
          row.names = FALSE)
writeLines('cat(1)', file.path(d, 'a.R'))
prov <- list(name = 'd', source = list(url = 'https://example.org', retrieved = '2026-10-20'),
             resource = list(filename = 'd.csv', sha256 = sha256file(file.path(d, 'd.csv')),
                             size_bytes = file.size(file.path(d, 'd.csv')), row_count_data_rows = 6),
             script = list(filename = 'a.R', sha256 = sha256file(file.path(d, 'a.R'))),
             schema = list(expected_columns = c('id', 'g'), expected_types = list(id = 'integer')))
writeLines(tojs(prov, auto_unbox = TRUE, pretty = TRUE), file.path(d, 'data_provenance.json'))
m <- mkmnf(list(project = 'p'), env = FALSE)
m <- suppressMessages(recrd(m, 'mean_x', obs = mean((1:6) / 7), expd = 0.5, tol = 1e-12))
m <- suppressMessages(recrd(m, 'n', obs = 6, expd = 6))
wrmnfjs(m, file.path(d, 'manifest.json'))
v <- vrfcps(d)
out <- list(ok = v$ok, check = v$checks$check, pass = v$checks$ok, digest = mnfdgst(m))
""",
    )
    py = cps.verify(d)
    assert out["ok"] is True and py["ok"] is True, py
    assert out["check"] == [c["check"] for c in py["checks"]]
    assert cps.digest(cps.read_manifest(d / "manifest.json")) == out["digest"]
    # and a tampered copy fails the same rows on both sides
    with open(d / "d.csv", "a") as fh:
        fh.write('7,"a",1\n')
    out = _r(tmp_path, f"v <- vrfcps('{d}'); out <- list(check = v$checks$check, pass = v$checks$ok)")
    assert list(zip(out["check"], out["pass"])) == _rows(cps.verify(d))


@needs_r
def test_r_checks_morie_hmac_bundle(tmp_path):
    d = _capsule(tmp_path / "c")
    m = cps.record(cps.manifest({"project": "p"}, env=False), "n", 5, 5)
    cps.write_manifest(m, d / "manifest.json")
    b = cps.bundle(d, m, KEY, note="from morie")
    a = b["attestation"]
    out = _r(
        tmp_path,
        f"""
b <- cpsbndlrd('{d}/capsule_bundle.json')
m <- fromjs('{d}/manifest.json', simplifyVector = FALSE)
haskey <- "key" %in% names(formals(cpsbndlvrf))
v <- if (haskey) cpsbndlvrf(b, '{d}', man = m, key = '{KEY}') else cpsbndlvrf(b, '{d}', man = m)
pl <- unclass(b$attestation)[c('digest','scheme','public','context','prehash','note','signed_utc')]
signed <- csha256(charToRaw(mnfcanon(pl)))
out <- list(check = v$checks$check, pass = v$checks$ok, detail = v$checks$detail, haskey = haskey,
            ok = v$ok, sig = cpsvrf(signed, b$attestation$signature, '{KEY}'),
            tag = chmac('{KEY}', signed), mdig = mnfdgst(m))
""",
    )
    rows = dict(zip(out["check"], out["pass"]))
    # R verifies every file, the listing, the attestation and manifest digests...
    assert all(ok for c, ok in rows.items() if c != "attestation:signature"), out
    # ...and the HMAC itself through its own cpsvrf()/chmac() over the same payload digest
    assert out["sig"] is True and out["tag"] == a["signature"]["signature"]
    assert out["mdig"] == b["manifest_digest"]
    if out["haskey"]:
        # rmoriebricklayer with cpsbndlvrf(key = ): the hmac attestation verifies with the shared key
        assert rows["attestation:signature"] is True and out["ok"] is True, out
    else:
        # older 1.0.0 builds have no hmac attestation key, so their signature row is a FAIL
        assert rows["attestation:signature"] is False


@needs_r
def test_morie_checks_r_mldsa_bundle(tmp_path):
    d = _capsule(tmp_path / "c")
    out = _r(
        tmp_path,
        f"""
m <- suppressMessages(recrd(mkmnf(list(project = 'p'), env = FALSE), 'n', obs = 5, expd = 5))
wrmnfjs(m, '{d}/manifest.json')
k <- fipskg('ML-DSA-44')
b <- cpsbndl('{d}', m, k, note = 'from R')
out <- list(ok = cpsbndlvrf('{d}/capsule_bundle.json', '{d}', man = m)$ok, public = k$public)
""",
    )
    assert out["ok"] is True
    m = cps.read_manifest(d / "manifest.json")
    chk = cps.verify_bundle(str(d / "capsule_bundle.json"), d, manifest=m, public_key=out["public"])
    rows = dict(_rows(chk))
    assert all(ok for c, ok in rows.items() if c != "attestation:signature"), chk
    assert rows["attestation:signature"] is False  # ML-DSA is verified by the R package, not here


@needs_r
def test_r_compares_morie_manifest(tmp_path):
    m = cps.manifest({"project": "p", "n": 3}, env=True)
    cps.record(m, "third", 1 / 3, 0.3333, tol=0.001)
    cps.record(m, "big", 1e20, 1e20)
    cps.record(m, "label", "CA", "CA")
    p = cps.write_manifest(m, tmp_path / "m.json")
    out = _r(
        tmp_path,
        f"""
m <- fromjs('{p}', simplifyVector = FALSE)
st <- lapply(names(m$results), function(nm) {{ force(nm); function(d) m$results[[nm]]$observed }})
names(st) <- names(m$results)
r <- mnfrcmp(m, NULL, st)
f <- tempfile(); wrmnfjs(m, f)
out <- list(ok = r$ok, status = r$results$status, digest = mnfdgst(m),
            pretty = paste(readLines(f), collapse = '\\n'))
""",
    )
    assert out["ok"] is True and set(out["status"]) == {"MATCH"}
    assert out["digest"] == cps.digest(m)
    assert out["pretty"] + "\n" == pathlib.Path(p).read_text()


@needs_r
def test_drift_matches_r(tmp_path):
    rng = random.Random(3)
    days = [f"2026-01-{i % 28 + 1:02d}" for i in range(300)]
    ref = _csv(
        tmp_path / "r.csv",
        ["x", "k", "g", "day", "id"],
        [[rng.gauss(0, 1), rng.randint(0, 9), rng.choice("abcd"), days[i], f"id{i}"] for i in range(300)],
    )
    cur = _csv(
        tmp_path / "c.csv",
        ["x", "k", "g", "day", "id"],
        [
            [rng.gauss(0.3, 1), rng.randint(0, 12), rng.choice("abcde"), days[(i * 7) % 300], f"id{i + 100}"]
            for i in range(250)
        ],
    )
    out = _r(
        tmp_path,
        f"""
r <- cpsdrft(read.csv('{ref}', stringsAsFactors = FALSE), read.csv('{cur}', stringsAsFactors = FALSE))
cl <- r$columns
out <- list(column = cl$column, type = cl$type, statistic = cl$statistic, p_value = cl$p_value, psi = cl$psi,
            jsdiv = cl$jsdiv, drifted = cl$drifted, anydrft = r$anydrft,
            benford = benfordtst(read.csv('{cur}')$x)$statistic)
""",
    )
    py = cps.drift(ref, cur)
    assert [c["column"] for c in py["columns"]] == out["column"]
    assert [c["type"] for c in py["columns"]] == out["type"]
    for i, c in enumerate(py["columns"]):
        for f in ("statistic", "p_value", "psi", "jsdiv"):
            want = out[f][i]
            assert (c[f] is None) == (want is None), (c["column"], f)
            if want is not None:
                assert c[f] == pytest.approx(want, rel=1e-9, abs=1e-12), (c["column"], f)
        assert c["drifted"] == out["drifted"][i]
    assert py["anydrft"] == out["anydrft"]
    xs = [float(r[0]) for r in list(csv.reader(pathlib.Path(cur).read_text().splitlines()))[1:]]
    assert cps.benford(xs)["statistic"] == pytest.approx(out["benford"], rel=1e-12)
