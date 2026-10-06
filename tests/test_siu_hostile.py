"""The SIU core is vendored from rmoriebricklayer, where 25 KB of whitespace
through the regex passes overflowed the C stack (its 0.5.7 review). The
passes over a whole document are loops now and every line the extractors
see is capped; this is the same guard the canonical package runs, so the
copy here cannot drift back."""

import pytest

native = pytest.importorskip("morie._core")

HOSTILE = [
    " " * 300_000,
    "a" * 300_000,
    "<" + "a" * 300_000,
    "&#" + "9" * 100_000 + ";",
    "\n" * 300_000,
    "<p>x" * 100_000,
    "<script>" + "x" * 300_000,
    " " * 200_000,
]


@pytest.mark.parametrize("text", HOSTILE, ids=[f"case{i}" for i in range(len(HOSTILE))])
def test_every_siu_binding_returns_on_hostile_text(text):
    assert isinstance(native.siu_html_to_text(text), str)
    assert isinstance(native.siu_parse_report_html(text), dict)
    assert isinstance(native.siu_parse_report_text(text), dict)
    count, reason = native.siu_resolve_so(text)
    assert isinstance(reason, str)


def test_caps_match_the_canonical_package():
    assert native.siu_to_iso_date("x" * 5000) == ""
    with pytest.raises(ValueError, match="larger than 2 MiB"):
        native.siu_html_to_text("a" * (3 * 1024 * 1024))
    out = native.siu_html_to_text("<p>" + " ".join(["word"] * 1500) + "</p>")
    assert all(len(line) <= 4000 for line in out.split("\n"))
