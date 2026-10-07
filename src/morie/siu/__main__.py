# SPDX-License-Identifier: AGPL-3.0-or-later
"""``python -m morie.siu`` -- the siu command line (same commands as the C++ ``siu`` binary and R ``morie_siu_cli()``).

  version
  models  [backend]
  chat    [backend] --model M PROMPT
  fetch   DRID
  parse   REPORT.html|.txt          native parser -> schema-field JSON
  resolve REPORT.txt                deterministic subject-official count
  audit   PARSED.json REPORT.txt [backend] [panel options]

Backend (any model; nothing hardcoded): --api ollama|openai, --base URL,
--key TOKEN, --timeout S, --temperature T (defaults from MORIE_LLM_* /
OLLAMA_* / OPENAI_* environment variables). Panel options: --mode 1-4,
--readers a,b, --auditors a,b, --num-readers N, --num-auditors M,
--reader-concurrency N, --auditor-sequential 0|1,
--reader-granularity all|per-field, --auditor-granularity all|per-field,
--no-health-check.
"""

from __future__ import annotations

import json
import sys

VERSION = "siu (morie) 1.0.0"


def _opt(args: list[str], name: str, default: str = "") -> str:
    for i in range(len(args) - 1):
        if args[i] == name:
            return args[i + 1]
    return default


def _csv(s: str) -> list[str] | None:
    return [t for t in s.split(",") if t] or None


def _slurp(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "siu":
        args = args[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = args[0]
    from . import llm
    from .audit import siu_audit_panel
    from .corpus import resolve_subject_officials
    from .native import html_to_text, parse_report_html, parse_report_text

    be = dict(
        api=_opt(args, "--api"),
        base=_opt(args, "--base"),
        key=_opt(args, "--key"),
        timeout=float(_opt(args, "--timeout", "300")),
        temperature=float(_opt(args, "--temperature", "0")),
    )
    try:
        if cmd == "version":
            print(VERSION)
            return 0
        if cmd == "models":
            b = llm.resolve(**be)
            ms = llm.list_models(b)
            print(f"server: {b.base} [{b.api}] ({len(ms)} models)")
            for m in ms:
                print(f"  {m}")
            return 0 if ms else 1
        if cmd == "chat" and len(args) >= 2:
            b = llm.resolve(**be)
            model = _opt(args, "--model") or llm.default_model(b)
            if not model:
                raise RuntimeError("no model: pass --model or set MORIE_LLM_MODEL")
            print(llm.chat(b, model, args[-1]))
            return 0
        if cmd == "fetch" and len(args) == 2:
            import urllib.request

            url = f"https://www.siu.on.ca/en/directors_report_details.php?drid={int(args[1])}"
            req = urllib.request.Request(url, headers={"User-Agent": "morie-siu/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                print(html_to_text(r.read().decode("utf-8", errors="replace")))
            return 0
        if cmd == "parse" and len(args) == 2:
            raw = _slurp(args[1])
            f = parse_report_html(raw) if "<" in raw else parse_report_text(raw)
            print(json.dumps(f, indent=1, ensure_ascii=False, sort_keys=True))
            return 0
        if cmd == "resolve" and len(args) == 2:
            n, why = resolve_subject_officials(_slurp(args[1]))
            print(f"subject_officers={'UNRESOLVED' if n is None else n}  ({why})")
            return 0 if n is not None else 1
        if cmd == "audit" and len(args) >= 3:
            res = siu_audit_panel(
                _slurp(args[2]),
                _slurp(args[1]),
                mode=int(_opt(args, "--mode", "4")),
                readers=_csv(_opt(args, "--readers")),
                auditors=_csv(_opt(args, "--auditors", _opt(args, "--auditor"))),
                num_readers=int(_opt(args, "--num-readers", "0")),
                num_auditors=int(_opt(args, "--num-auditors", "0")),
                reader_concurrency=int(_opt(args, "--reader-concurrency", "0")),
                auditor_sequential=_opt(args, "--auditor-sequential", "1") != "0",
                reader_granularity=_opt(args, "--reader-granularity", "all"),
                auditor_granularity=_opt(args, "--auditor-granularity", "all"),
                health_check="--no-health-check" not in args,
                **be,
            )
            print(res["json"])
            return 0
    except Exception as e:  # noqa: BLE001 -- CLI boundary: report and exit non-zero
        print(f"siu: {e}", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
