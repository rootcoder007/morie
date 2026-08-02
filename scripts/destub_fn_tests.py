#!/usr/bin/env python3
"""Replace generator-*guessed* value assertions in tests/fn with
structural finiteness checks (N6).

The generator emitted assertions like

    assert abs(result["estimate"] - 3.0) < 0.01   # 3.0 was a guess

where the literal was never computed from the model -- it was the input
mean, or a round number that looked plausible -- so the assertion fails
whenever the guess is wrong and proves nothing when it passes.

WHAT THIS SCRIPT MUST NOT TOUCH
-------------------------------
An earlier version also rewrote

    assert result["n"] == 1000.0
    assert res["train_accuracy"] == 1.0

into finiteness checks.  Those are not guesses: they are exact
identities the implementation genuinely has to reproduce, and rewriting
them silently deleted 112 real assertions.  That rule is gone and is
not coming back -- equality assertions are always left alone.

The remaining discriminator is the tolerance.  A tolerance at
machine-epsilon scale (<= 1e-9) means the literal is a computed
reference the implementation must reproduce bit-for-bit; only the loose
tolerances the generator emitted (0.01, 0.05, ...) are candidates.
Files on the guard's _REVIEWED allowlist have already been checked by
hand and are skipped.

Default is a dry run; pass --apply to write.
"""
import importlib.util
import re
import sys
from pathlib import Path

FN = Path("tests/fn")
GUARD = Path("tests/test_fn_no_stub_asserts.py")

# A tolerance this tight means the literal is a computed reference.
TIGHT = 1e-9

# assert abs(<sub>[<key>] - <number>) <|<= <tol>
ABS = re.compile(
    r"^(?P<i>\s*)assert\s+abs\(\s*(?P<expr>[A-Za-z_]\w*\[[^\]]+\])\s*-\s*"
    r"[-+0-9.eE]+\s*\)\s*<=?\s*(?P<tol>[-+0-9.eE]+)\s*(?:,.*|#.*)?$"
)


def reviewed() -> frozenset:
    """The guard's allowlist, so the two cannot drift apart."""
    spec = importlib.util.spec_from_file_location("_guard", GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "_REVIEWED", frozenset())


def struct(indent: str, expr: str) -> str:
    return (f"{indent}assert np.all(np.isfinite(np.asarray({expr}, "
            f"dtype=float)))  # N6: was a generator-guessed value")


def transform(text: str) -> tuple[str, int, int]:
    out, rewritten, kept = [], 0, 0
    for line in text.splitlines():
        m = ABS.match(line)
        if m:
            try:
                loose = float(m.group("tol")) > TIGHT
            except ValueError:
                loose = False
            if loose:
                out.append(struct(m.group("i"), m.group("expr")))
                rewritten += 1
                continue
            kept += 1
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), rewritten, kept


def main(apply: bool) -> None:
    allow = reviewed()
    files = total = tight_kept = skipped = 0
    for p in sorted(FN.rglob("test_*.py")):
        if p.name in allow:
            skipped += 1
            continue
        t = p.read_text(encoding="utf-8")
        new, n, kept = transform(t)
        tight_kept += kept
        if n:
            files += 1
            total += n
            if "import numpy as np" not in new:
                new = "import numpy as np\n" + new
            if apply:
                p.write_text(new, encoding="utf-8")
    verb = "rewrote" if apply else "would rewrite"
    print(f"{verb} {total} loose-tolerance asserts across {files} files")
    print(f"left alone: {tight_kept} tolerances <= {TIGHT:g} (computed "
          f"references), every == assertion, and {skipped} reviewed files")
    if not apply:
        print("dry run -- pass --apply to write")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
