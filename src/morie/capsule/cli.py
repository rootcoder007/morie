# SPDX-License-Identifier: AGPL-3.0-or-later
"""``morie capsule init | pin | verify | keygen | sign | drift``.

Exit codes as the rmbl-action: 0 everything verified (no drift), 1 a check
failed or drift was found, 2 a usage or input error. ``--json`` prints the
result as JSON instead of lines. A signing key is never taken on the
command line: it is read from an environment variable (``--key-env``,
default ``MORIE_CAPSULE_KEY``) or a file (``--key-file``). For ``--scheme
hmac`` (the default) that is the shared secret; for ``ML-DSA-44``,
``ML-DSA-65``, ``ML-DSA-87`` and ``xmss-sha256`` it is a JSON key file from
``morie capsule keygen`` (or an R key object written with ``tojs()``). An
XMSS key must come from ``--key-file``: its ``next_index`` is written back
after every signature, and the used-index record shared with
rmoriebricklayer refuses an index already used.

    morie capsule keygen --scheme ML-DSA-65 --out key.json
    morie capsule sign DIR --scheme ML-DSA-65 --key-file key.json
    morie capsule verify DIR --public-key <hex printed by keygen>
"""

from __future__ import annotations

import json
import os
import sys

from . import _csv, _drift, _manifest, _pq, _prov, _sign

KEY_ENV = "MORIE_CAPSULE_KEY"


def register_subparser(subparsers) -> None:
    """Called from morie.runner.build_parser() to register ``morie capsule``."""
    p = subparsers.add_parser(
        "capsule",
        help="Author, pin, verify, sign and drift-check data capsules (the rmoriebricklayer / rmbl-action format)",
    )
    sub = p.add_subparsers(dest="capsule_cmd", required=True)

    def common(q):
        q.add_argument("--json", action="store_true", help="print the result as JSON")

    def keyargs(q):
        q.add_argument(
            "--key-env",
            default=KEY_ENV,
            help=f"environment variable holding the HMAC key or a JSON ML-DSA key (default {KEY_ENV})",
        )
        q.add_argument(
            "--key-file",
            default=None,
            help="file holding the HMAC key (its bytes, final newline dropped) or a JSON key from `capsule keygen`",
        )

    q = sub.add_parser("init", help="pin a CSV: write data_provenance.json beside it")
    q.add_argument("csv")
    q.add_argument("--url", required=True, help="where the data came from (source.url)")
    q.add_argument("--name", default=None)
    q.add_argument("--retrieved", default=None, help="YYYY-MM-DD (default today)")
    q.add_argument("--script", default=None, help="analysis script beside the CSV to pin (relative to its directory)")
    common(q)

    q = sub.add_parser("pin", help="re-pin a capsule's provenance to its files as they are now")
    q.add_argument("dir")
    common(q)

    q = sub.add_parser("verify", help="verify a capsule (provenance, manifest, script) and its bundle if present")
    q.add_argument("dir")
    q.add_argument(
        "--public-key",
        default=None,
        help="expected attestation key (hex) of the bundle: the public key `capsule keygen` printed, or an hmac key id",
    )
    keyargs(q)
    common(q)

    q = sub.add_parser(
        "keygen", help="write a new ML-DSA or XMSS signing key (JSON, owner-only) and print its public key"
    )
    q.add_argument("--scheme", required=True, choices=_pq.SCHEMES)
    q.add_argument("--out", required=True, help="where to write the signing key (SECRET; must not exist)")
    q.add_argument("--height", type=int, default=10, help="xmss-sha256 tree height, 1-16: 2^height signatures")
    q.add_argument("--public-out", default=None, help="also write the public key (JSON) here")
    common(q)

    q = sub.add_parser("sign", help="write capsule_bundle.json, signed with an HMAC, ML-DSA or XMSS key")
    q.add_argument("dir")
    q.add_argument("--note", default=None, help="signer's note, covered by the signature")
    q.add_argument(
        "--scheme",
        default="hmac",
        choices=("hmac", *_pq.SCHEMES),
        help="hmac (shared secret, the default), ML-DSA-44/65/87 or xmss-sha256 (a key from `capsule keygen`)",
    )
    q.add_argument("--context", default=None, help="ML-DSA context string, bound into the signature")
    q.add_argument("--prehash", default="none", help="ML-DSA pre-hash: none (default), sha256, sha512, ...")
    keyargs(q)
    common(q)

    q = sub.add_parser(
        "drift", help="compare a capsule's pinned CSV (or a CSV) with a current one: KS, chi-square, PSI, JS, Benford"
    )
    q.add_argument("ref", help="capsule directory or reference CSV")
    q.add_argument("cur")
    q.add_argument("--alpha", type=float, default=0.01)
    q.add_argument("--psi-threshold", type=float, default=0.25)
    common(q)


def _key(args, required: bool):
    if args.key_file:
        with open(args.key_file, "rb") as fh:
            k = fh.read()
        return k[:-1] if k.endswith(b"\n") else k
    k = os.environ.get(args.key_env)
    if not k and required:
        raise ValueError(f"no key: set {args.key_env} or pass --key-file")
    return k or None


def _short(h: str) -> str:
    return h if len(h) <= 64 else f"{h[:32]}... ({len(h) // 2} bytes)"


def _lines(rows) -> list[str]:
    return [f"  {'ok  ' if r['ok'] else 'FAIL'}  {r['check']:<26} {r['detail']}" for r in rows]


def _out(args, obj, lines: list[str]) -> None:
    if args.json:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print("\n".join(lines))


def _verify(args) -> int:
    d = args.dir
    if not os.path.isdir(d):
        raise ValueError(f"not a directory: {d}")
    out, lines, ok = {}, [], True
    have_prov = os.path.exists(os.path.join(d, _prov.PROVFILE))
    have_bundle = os.path.exists(os.path.join(d, _sign.BUNDLE))
    if not have_prov and not have_bundle:
        raise ValueError(f"{d} holds neither {_prov.PROVFILE} nor {_sign.BUNDLE}")
    if have_prov:
        c = _prov.verify(d)
        out["capsule"] = c
        ok = ok and c["ok"]
        lines += [f"Capsule verification: {'intact' if c['ok'] else 'NOT VERIFIED'}", *_lines(c["checks"])]
        lines += [f"  FAIL  {m:<26} required check was never made" for m in c["missing"]]
    if have_bundle:
        b = _sign.verify_bundle(os.path.join(d, _sign.BUNDLE), d, key=_key(args, False), public_key=args.public_key)
        out["bundle"] = b
        ok = ok and b["ok"]
        lines += [f"Bundle check: {'OK' if b['ok'] else 'FAILED'}", *_lines(b["checks"])]
    out["ok"] = ok
    _out(args, out, lines)
    return 0 if ok else 1


def _sign_cmd(args) -> int:
    d = args.dir
    if not os.path.isdir(d):
        raise ValueError(f"not a directory: {d}")
    mpath = os.path.join(d, "manifest.json")
    man = _manifest.read_manifest(mpath) if os.path.exists(mpath) else _manifest.manifest(env=False)
    if args.scheme == "hmac":
        if args.context is not None or args.prehash != "none":
            raise ValueError("--context and --prehash are for ML-DSA; hmac takes neither")
        b = _sign.bundle(d, man, _key(args, True), note=args.note)
        print(
            "note: the hmac signature needs the shared key: rmoriebricklayer >= 1.0.0 (srr) verifies it with"
            " cpsbndlvrf(key = ...), and `morie capsule verify --key-env ...` checks it",
            file=sys.stderr,
        )
    else:
        if args.scheme == _pq.XMSS and not args.key_file:
            raise ValueError("an xmss-sha256 key must come from --key-file: its next_index is written back to it")
        key = _pq.load_key(_key(args, True))
        if key["scheme"] != args.scheme:
            raise ValueError(f"--scheme {args.scheme} but the key is {key['scheme']}")
        b = _sign.bundle(d, man, key, note=args.note, context=args.context, prehash=args.prehash)
        if args.scheme == _pq.XMSS:
            _pq.write_key(key, args.key_file)
    a = b["attestation"]
    _out(
        args,
        b,
        [
            f"wrote {os.path.join(d, _sign.BUNDLE)}: {len(b['files'])} files, scheme {a['scheme']}, key {_short(a['public'])}"
        ],
    )
    return 0


def _keygen_cmd(args) -> int:
    if os.path.exists(args.out):
        raise ValueError(f"{args.out} exists: a signing key is never overwritten")
    key = _pq.keygen(args.scheme, args.height)
    _pq.write_key(key, args.out)
    pub = _pq.public_key(key)
    if args.public_out:
        with open(args.public_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(pub, indent=2) + "\n")
    hexkey = _pq.public_hex(key)
    _out(
        args,
        {"scheme": key["scheme"], "public": hexkey, "key_file": args.out},
        [f"wrote {args.out} ({key['scheme']}); public key:", hexkey],
    )
    return 0


def _drift_cmd(args) -> int:
    r = _drift.drift(args.ref, args.cur, alpha=args.alpha, psi_threshold=args.psi_threshold)
    cur = _csv.frame(args.cur)
    ben = {}
    for nm, (typ, vals) in cur.items():
        if typ in ("integer", "numeric"):
            try:
                b = _drift.benford(vals)
            except ValueError:
                continue
            ben[nm] = {k: b[k] for k in ("statistic", "df", "p_value", "n")}
    r["benford"] = ben

    def f(v):
        return "NA" if v is None else f"{v:.4g}"

    lines = [f"Drift: {'DRIFT' if r['anydrft'] else 'none'} ({r['nref']} -> {r['ncur']} rows)"]
    for c in r["columns"]:
        flag = "DRIFT" if c["drifted"] is True else "ok   " if c["drifted"] is False else "--   "
        lines.append(
            f"  {flag} {c['column']:<20} {c['type']:<11} stat {f(c['statistic'])}  p {f(c['p_value'])}"
            f"  psi {f(c['psi'])}  js {f(c['jsdiv'])}  {c['note']}"
        )
    lines += [f"  added   {a}" for a in r["added"]] + [f"  removed {a}" for a in r["removed"]]
    lines += [f"  benford {k:<20} chi2 {f(v['statistic'])}  p {f(v['p_value'])}  n {v['n']}" for k, v in ben.items()]
    _out(args, r, lines)
    return 1 if r["anydrft"] else 0


def run(args) -> int:
    """Run ``morie capsule <cmd>``; returns the exit code (0 ok, 1 failed/drift, 2 error)."""
    try:
        if args.capsule_cmd == "init":
            d = os.path.dirname(os.path.abspath(args.csv))
            p = _prov.write(
                d,
                os.path.basename(args.csv),
                args.url,
                name=args.name,
                retrieved=args.retrieved,
                script_file=args.script,
            )
            path = os.path.join(d, _prov.PROVFILE)
            _out(args, p, [f"wrote {path}: {p['resource']['filename']} sha256 {p['resource']['sha256']}"])
            return 0
        if args.capsule_cmd == "pin":
            p = _prov.pin(args.dir)
            _out(args, p, [f"pinned {p['resource']['filename']} sha256 {p['resource']['sha256']}"])
            return 0
        if args.capsule_cmd == "verify":
            return _verify(args)
        if args.capsule_cmd == "keygen":
            return _keygen_cmd(args)
        if args.capsule_cmd == "sign":
            return _sign_cmd(args)
        if args.capsule_cmd == "drift":
            return _drift_cmd(args)
    except (OSError, ValueError, TypeError) as e:
        print(f"morie capsule {args.capsule_cmd}: {e}", file=sys.stderr)
        return 2
    return 2
