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
    # a 5000-byte "date" is an error, as in the canonical package (0.5.9), never a silent ""
    with pytest.raises(ValueError, match="longer than 4096 bytes"):
        native.siu_to_iso_date("x" * 5000)
    with pytest.raises(ValueError, match="larger than 2 MiB"):
        native.siu_html_to_text("a" * (3 * 1024 * 1024))
    out = native.siu_html_to_text("<p>" + " ".join(["word"] * 1500) + "</p>")
    assert all(len(line) <= 4000 for line in out.split("\n"))


def test_absurd_tag_number_is_noise_not_a_crash():
    # std::stoi on a \d+ capture past INT_MAX used to abort the process (fuzzer, 2026-10-06)
    txt = "Subject Officials\nSO\n#" + "4" * 30 + "\nCivilian Witnesses\nCW #2\n"
    f = native.siu_parse_report_text(txt)
    assert f["number_of_subject_officials"] == "1"
    assert f["number_of_civilian_witnesses"] == "2"
    count, reason = native.siu_resolve_so(txt)
    assert isinstance(reason, str)


@pytest.mark.parametrize("text", [
    "SO #1" + "\f" * 25000 + "x",                                   # form feeds (0.5.8 diff review)
    "SO #1" + "\v" * 25000 + "x",
    "this information may include" + "a b\n" * 15000 + "affected person.",   # boilerplate run (SIU review)
    "who, in the opinion of the SIU Director, is not a subject officer " + "a b\n" * 12500,
], ids=["formfeed", "vtab", "boilerplate", "glossary"])
def test_the_0_5_8_reviews_shapes_return(text):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(native.siu_parse_report_text(text), dict)
        count, reason = native.siu_resolve_so(text)
        assert isinstance(reason, str)


def test_long_lines_are_split_with_a_warning_and_the_scans_are_linear():
    import time, warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        native.siu_html_to_text("<p>" + "word " * 1000 + "</p>")
    assert any("longer than 2000" in str(x.message) for x in w)
    t0 = time.perf_counter()
    native.siu_html_to_text("<script>" * 40000)
    assert time.perf_counter() - t0 < 5
    with pytest.raises(ValueError, match="longer than 4096 bytes"):
        native.siu_to_iso_date("x" * 5000)
