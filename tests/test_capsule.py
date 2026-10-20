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
    assert chk["ok"] is True, chk  # morie verifies ML-DSA itself


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


# ------------------------------------------------------------------ post-quantum schemes


@pytest.fixture
def xrec(tmp_path, monkeypatch):
    """A private XMSS used-index record (shared by morie and, through the env, by R)."""
    p = tmp_path / "rec" / "xmss_used_indices.txt"
    monkeypatch.setenv("RMBL_XMSS_INDEX_FILE", str(p))
    return p


PQ = ["ML-DSA-44", "xmss-sha256"]


def _pqkey(scheme):
    return cps.keygen(scheme, height=3)


def _tamper_sig(path):
    b = json.loads(path.read_text())
    s = b["attestation"]["signature"]["signature"]
    b["attestation"]["signature"]["signature"] = ("0" if s[0] != "0" else "1") + s[1:]
    path.write_text(json.dumps(b, indent=2))


@pytest.mark.parametrize("scheme", PQ)
def test_pq_sign_verify(scheme, xrec):
    k = _pqkey(scheme)
    pub = cps.public_key(k)
    assert "secret" not in pub and "skseed" not in pub and "skprf" not in pub
    s = cps.sign("a manifest digest", k)
    assert s["scheme"] == scheme
    assert cps.verify_signature("a manifest digest", s, pub)
    assert cps.verify_signature("a manifest digest", s, k)
    assert not cps.verify_signature("a manifest digesT", s, pub)
    assert not cps.verify_signature("a manifest digest", s, cps.public_key(_pqkey(scheme)))
    bad = dict(s, signature=("0" if s["signature"][0] != "0" else "1") + s["signature"][1:])
    assert not cps.verify_signature("a manifest digest", bad, pub)
    assert not cps.verify_signature("a manifest digest", {"scheme": "hmac", "signature": "00"}, pub)
    with pytest.raises(ValueError, match="does not match"):
        cps.sign("m", k, scheme="hmac")
    with pytest.raises(ValueError, match="needs a key from keygen"):
        cps.sign("m", "secret", scheme=scheme)
    if scheme == "xmss-sha256":
        assert list(s) == ["scheme", "signature", "auth", "index", "root", "height", "randomizer", "wire"]
        assert s["index"] == 0 and k["next_index"] == 1
    else:
        assert list(s) == ["scheme", "signature", "context", "prehash"]
        c = cps.sign("m", k, context="release", prehash="sha512")
        assert c["context"] == "cmVsZWFzZQ==" and c["prehash"] == "sha512"
        assert cps.verify_signature("m", c, pub, context="release")
        assert not cps.verify_signature("m", c, pub, context="staging")
        assert not cps.verify_signature("m", c, pub, context="release", prehash="none")


def test_xmss_index_never_reused(xrec, tmp_path):
    k = cps.keygen("xmss-sha256", height=2)
    assert xrec.read_text().endswith("\t-1\n")
    stale = dict(k)
    cps.sign("one", k)
    with pytest.raises(ValueError, match="index 0 of this key was already used"):
        cps.sign("two", stale)  # a stale copy of the key
    cps.write_key(k, tmp_path / "k.json")
    k2 = cps.read_key(tmp_path / "k.json")
    assert k2["next_index"] == 1 and cps.sign("two", k2)["index"] == 1
    for _ in range(2):
        cps.sign("more", k2)
    with pytest.raises(ValueError, match="exhausted"):
        cps.sign("five", k2)
    # a key the record does not know may not sign from index 0
    k3 = dict(cps.keygen("xmss-sha256", height=1), skseed="ab" * 32)
    with pytest.raises(ValueError, match="not in the used-index record"):
        cps.sign("x", k3)
    lines = xrec.read_text().splitlines()
    assert all(ln.startswith("k:") and len(ln.split("\t")[0]) == 66 for ln in lines)


@pytest.mark.parametrize("scheme", PQ)
def test_pq_bundle_roundtrip(scheme, xrec, tmp_path):
    d = _capsule(tmp_path / "c")
    m = cps.record(cps.manifest({"project": "p"}, env=False), "n", 5, 5)
    cps.write_manifest(m, d / "manifest.json")
    k = _pqkey(scheme)
    b = cps.bundle(d, m, k, note="pq")
    a = b["attestation"]
    assert a["scheme"] == scheme and a["public"] == cps.key_id(k)
    path = d / "capsule_bundle.json"
    assert cps.verify_bundle(str(path), d, manifest=m, public_key=cps.public_key(k))["ok"]
    other = _pqkey(scheme)
    assert not cps.verify_bundle(str(path), d, public_key=cps.key_id(other))["ok"]
    # the attestation names another key: its signature no longer verifies
    j = json.loads(path.read_text())
    j["attestation"]["public"] = cps.key_id(other)
    path.write_text(json.dumps(j))
    assert dict(_rows(cps.verify_bundle(str(path), d)))["attestation:signature"] is False
    cps.bundle(d, m, k)
    _tamper_sig(path)
    assert dict(_rows(cps.verify_bundle(str(path), d)))["attestation:signature"] is False
    with pytest.raises(ValueError, match="hmac attestation takes no"):
        cps.attest(m, KEY, context="x")


@pytest.mark.parametrize("scheme", PQ)
def test_pq_cli(scheme, xrec, tmp_path, capsys, monkeypatch):
    from morie.capsule.cli import run
    from morie.runner import build_parser

    def cli(*a):
        return run(build_parser().parse_args(["capsule", *a]))

    d = tmp_path / "c"
    d.mkdir()
    f = _csv(d / "d.csv", ["id"], [[1], [2]])
    assert cli("init", f, "--url", "https://example.org") == 0
    capsys.readouterr()
    kf = tmp_path / "key.json"
    assert cli("keygen", "--scheme", scheme, "--height", "3", "--out", str(kf), "--json") == 0
    pub = json.loads(capsys.readouterr().out)["public"]
    assert oct(kf.stat().st_mode & 0o777) == "0o600" or os.name == "nt"
    assert cli("keygen", "--scheme", scheme, "--out", str(kf)) == 2  # never overwritten
    assert cli("sign", str(d), "--scheme", scheme, "--key-file", str(kf)) == 0
    assert cli("verify", str(d), "--public-key", pub) == 0
    assert cli("verify", str(d), "--public-key", "00" * 32) == 1
    assert (
        cli("sign", str(d), "--scheme", "ML-DSA-65" if scheme != "ML-DSA-65" else "ML-DSA-44", "--key-file", str(kf))
        == 2
    )
    if scheme == "xmss-sha256":
        assert json.loads(kf.read_text())["next_index"] == 1
        monkeypatch.setenv("MORIE_CAPSULE_KEY", kf.read_text())
        assert cli("sign", str(d), "--scheme", scheme) == 2  # state needs a file
        assert cli("sign", str(d), "--scheme", scheme, "--key-file", str(kf)) == 0
        assert json.loads(kf.read_text())["next_index"] == 2
    else:
        monkeypatch.setenv("MORIE_CAPSULE_KEY", kf.read_text())
        assert cli("sign", str(d), "--scheme", scheme, "--context", "release", "--prehash", "sha256") == 0
        assert cli("verify", str(d), "--public-key", pub) == 0
    _tamper_sig(d / "capsule_bundle.json")
    assert cli("verify", str(d)) == 1


# ------------------------------------------------------------------ post-quantum, cross-checks with R


@needs_r
def test_xmss_bytes_match_r(xrec, tmp_path):
    from morie.crypto import _xmss

    s1, s2, s3 = "11" * 32, "22" * 32, "33" * 32
    out = _r(
        tmp_path,
        f"""
k <- pqckg(3, '{s1}', '{s2}', '{s3}')
xmssregkey(k)
s <- cpssign('a manifest digest', k)
out <- list(root = k$root, wire = s$wire, sig = s$signature, auth = s$auth, r = s$randomizer)
""",
    )
    b1, b2, b3 = (bytes.fromhex(x) for x in (s1, s2, s3))
    assert _xmss.keygen_root(b1, b2, 3).hex() == out["root"]
    mine = _xmss.sign(b1, b3, b2, 3, 0, b"a manifest digest")
    assert (mine["wire"], mine["signature"], mine["auth"], mine["randomizer"]) == (
        out["wire"],
        out["sig"],
        out["auth"],
        out["r"],
    )
    # and the shared record now refuses index 0 of that key in morie too
    k = {
        "root": out["root"],
        "pubseed": s2,
        "skseed": s1,
        "skprf": s3,
        "height": 3,
        "next_index": 0,
        "capacity": 8,
        "scheme": "xmss-sha256",
    }
    with pytest.raises(ValueError, match="index 0 of this key was already used"):
        cps.sign("another message", k)


def _r_bundle_check(tmp_path, d, keyexp):
    return _r(
        tmp_path,
        f"""
v <- cpsbndlvrf('{d}/capsule_bundle.json', '{d}', keyexp = '{keyexp}')
out <- list(ok = v$ok, check = v$checks$check, pass = v$checks$ok)
""",
    )


@needs_r
@pytest.mark.parametrize("scheme", PQ)
def test_r_verifies_morie_pq_bundle(scheme, xrec, tmp_path):
    d = _capsule(tmp_path / "c")
    m = cps.record(cps.manifest({"project": "p"}, env=False), "n", 5, 5)
    cps.write_manifest(m, d / "manifest.json")
    k = _pqkey(scheme)
    if scheme == "xmss-sha256":
        cps.bundle(d, m, k, note="from morie")
    else:
        cps.bundle(d, m, k, note="from morie", context="release 2026-10", prehash="sha512")
    pub = cps.key_id(k)
    out = _r_bundle_check(tmp_path, d, pub)
    assert out["ok"] is True, out
    assert list(zip(out["check"], out["pass"])) == _rows(
        cps.verify_bundle(str(d / "capsule_bundle.json"), d, public_key=pub)
    )
    # wrong key: R refuses the pin, and the signature row is still good
    out = _r_bundle_check(tmp_path, d, cps.key_id(_pqkey(scheme)))
    rows = dict(zip(out["check"], out["pass"]))
    assert out["ok"] is False and rows["attestation:key_as_expected"] is False
    # a tampered signature byte fails on both sides
    _tamper_sig(d / "capsule_bundle.json")
    out = _r_bundle_check(tmp_path, d, pub)
    assert out["ok"] is False and dict(zip(out["check"], out["pass"]))["attestation:signature"] is False
    assert dict(_rows(cps.verify_bundle(str(d / "capsule_bundle.json"), d)))["attestation:signature"] is False


@needs_r
@pytest.mark.parametrize("scheme", PQ)
def test_morie_verifies_r_pq_bundle(scheme, xrec, tmp_path):
    d = _capsule(tmp_path / "c")
    keyc = "pqckg(3)" if scheme == "xmss-sha256" else f"fipskg('{scheme}')"
    extra = "" if scheme == "xmss-sha256" else ", ctx = 'release 2026-10', prehash = 'sha512'"
    out = _r(
        tmp_path,
        f"""
m <- suppressMessages(recrd(mkmnf(list(project = 'p'), env = FALSE), 'n', obs = 5, expd = 5))
wrmnfjs(m, '{d}/manifest.json')
k <- {keyc}
b <- cpsbndl('{d}', m, k, note = 'from R'{extra})
v <- cpsbndlvrf('{d}/capsule_bundle.json', '{d}', man = m)
pub <- if (inherits(k, 'bricklayer_signing_key')) paste0(k$root, k$pubseed) else k$public
other <- {keyc}
op <- if (inherits(other, 'bricklayer_signing_key')) paste0(other$root, other$pubseed) else other$public
out <- list(ok = v$ok, public = pub, other = op)
""",
    )
    assert out["ok"] is True
    path = str(d / "capsule_bundle.json")
    m = cps.read_manifest(d / "manifest.json")
    chk = cps.verify_bundle(path, d, manifest=m, public_key=out["public"])
    assert chk["ok"] is True, chk
    # wrong key
    rows = dict(_rows(cps.verify_bundle(path, d, public_key=out["other"])))
    assert rows["attestation:key_as_expected"] is False and rows["attestation:signature"] is True
    j = json.loads(pathlib.Path(path).read_text())
    j["attestation"]["public"] = out["other"]
    pathlib.Path(path + ".other").write_text(json.dumps(j))
    assert dict(_rows(cps.verify_bundle(path + ".other", d)))["attestation:signature"] is False
    os.remove(path + ".other")
    # tampered data and a tampered signature byte
    with open(d / "d.csv", "a") as fh:
        fh.write("9,a,1\n")
    assert dict(_rows(cps.verify_bundle(path, d)))["file:d.csv"] is False
    _tamper_sig(pathlib.Path(path))
    assert dict(_rows(cps.verify_bundle(path, d)))["attestation:signature"] is False
