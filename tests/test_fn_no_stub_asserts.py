"""Guard: tests/fn must never regain generator-guessed hardcoded-value asserts.

The 36k tests/fn suite was auto-generated once; a batch shipped assertions like
`assert abs(result["estimate"] - 3.0) < 0.01` where 3.0 was a guess (the input
mean), which fail whenever the guess is wrong. They were replaced with
structural finiteness checks. This test fails if that pattern comes back, so a
re-dump of generated tests can't silently reintroduce false expectations.
Run `python scripts/destub_fn_tests.py --apply` to clean any that appear.
"""
import re
from pathlib import Path

_FN = Path(__file__).resolve().parent / "fn"
# assert abs(<var>[<key>] - <number>) <|<= <number>   OR   <var>[<key>] == <float>
_BOGUS = re.compile(
    r"assert\s+abs\(\s*[A-Za-z_]\w*\[[^\]]+\]\s*-\s*[-+0-9.eE]+\s*\)\s*<=?\s*[-+0-9.eE]+\s*(?:,|#|$)"
    r"|assert\s+[A-Za-z_]\w*\[[^\]]+\]\s*==\s*[-+]?[0-9]+\.[0-9]+"
)


def test_no_generator_guessed_value_asserts():
    offenders = []
    for p in _FN.rglob("test_*.py"):
        if _BOGUS.search(p.read_text()):
            offenders.append(p.name)
    assert not offenders, (
        f"{len(offenders)} tests/fn file(s) have hardcoded-value asserts "
        f"(generator guesses). Run scripts/destub_fn_tests.py --apply. "
        f"First few: {offenders[:5]}"
    )
