"""Guard: tests/fn must never regain generator-guessed hardcoded-value asserts.

The 36k tests/fn suite was produced externally, once; a batch shipped
assertions like `assert abs(result["estimate"] - 3.0) < 0.01` where 3.0 was a
guess (the input mean), which fail whenever the guess is wrong. They were
replaced with structural finiteness checks. This test fails if that pattern
comes back, so a re-dump of generated tests can't silently reintroduce false
expectations. Clean any that appear with:

    python scripts/destub_fn_tests.py --apply

Implementation notes: the scan reads ~36k files, so it works on raw bytes --
no per-file decode (much faster, and immune to the locale-encoding trap where
Windows would read UTF-8 content as cp1252). It is a repository-content check,
identical on every platform, so it runs on POSIX only rather than paying the
very slow Windows small-file I/O cost in every matrix leg.
"""

import re
import sys
from pathlib import Path

import pytest

_FN = Path(__file__).resolve().parent / "fn"

# assert abs(<var>[<key>] - <number>) <|<= <number literal>   (a guessed point
# value against a literal tolerance), or  <var>[<key>] == <float literal>.
# Comparisons against a COMPUTED bound -- e.g.
#   assert abs(ate - 3) < abs(naive - 3) + 1.5
# -- are legitimate relative-accuracy tests and must NOT match.
_BOGUS = re.compile(
    rb"assert\s+abs\(\s*[A-Za-z_]\w*\[[^\]]+\]\s*-\s*[-+0-9.eE]+\s*\)\s*<=?\s*[-+0-9.eE]+\s*(?:,|#|\r?\n)"
    rb"|assert\s+[A-Za-z_]\w*\[[^\]]+\]\s*==\s*[-+]?[0-9]+\.[0-9]+"
)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="repo-content check; identical on every platform, and a 36k-file "
    "walk is pathologically slow on Windows I/O",
)
def test_no_generator_guessed_value_asserts():
    offenders = []
    for p in _FN.rglob("test_*.py"):
        if _BOGUS.search(p.read_bytes()):
            offenders.append(p.name)
    assert not offenders, (
        f"{len(offenders)} tests/fn file(s) have generator-guessed "
        f"hardcoded-value asserts. Run `python scripts/destub_fn_tests.py "
        f"--apply`. First few: {offenders[:5]}"
    )
