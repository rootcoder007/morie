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

# The generator's signature is the point-value-with-tolerance form:
#   assert abs(result["estimate"] - 3.0) < 0.01
# where the literal was a guess. Two forms deliberately do NOT match:
# a comparison against a COMPUTED bound (e.g. abs(ate - 3) < abs(naive
# - 3) + 1.5) is a legitimate relative-accuracy test, and a bare
# `x[key] == <float>` is an exact identity the maths guarantees or a
# value checked against a cited source (a Matern smoothness nu == 1.5,
# an LIL bound == 0.5 from eq. (2.21), an exact count). Matching the
# latter pushed real assertions out in favour of vacuous finiteness
# checks, so the scan is limited to the tolerance form.
_BOGUS = re.compile(
    rb"assert\s+abs\(\s*[A-Za-z_]\w*\[[^\]]+\]\s*-\s*[-+0-9.eE]+\s*\)"
    rb"\s*<=?\s*(?P<tol>[-+0-9.eE]+)\s*(?:,|#|\r?\n)"
)

# A tolerance at machine-epsilon scale (1e-9 or tighter) means the
# literal is a computed reference the implementation must reproduce
# bit-for-bit, not a guess -- only the loose tolerances the generator
# emitted (0.01, 0.05, ...) are flagged.
_TIGHT = 1e-9


def _is_guess(match) -> bool:
    try:
        return float(match.group("tol")) > _TIGHT
    except (TypeError, ValueError):
        return True


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="repo-content check; identical on every platform, and a 36k-file "
    "walk is pathologically slow on Windows I/O",
)
def test_no_generator_guessed_value_asserts():
    offenders = []
    for p in _FN.rglob("test_*.py"):
        if any(_is_guess(m) for m in
               _BOGUS.finditer(p.read_bytes())):
            offenders.append(p.name)
    assert not offenders, (
        f"{len(offenders)} tests/fn file(s) have generator-guessed "
        f"hardcoded-value asserts. Run `python scripts/destub_fn_tests.py "
        f"--apply`. First few: {offenders[:5]}"
    )
