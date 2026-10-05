"""The compiled SIU core (morie._core) and its pure-Python twin return the same fields.

Round-8 findings: the Python port had drifted from rmoriebricklayer's C++ (role-section
witness counts, French rosters, relative incident dates). Every case runs on both backends.
"""

import pytest

from morie.siu import native


@pytest.fixture(params=["compiled", "python"])
def backend(request, monkeypatch):
    if request.param == "compiled":
        if native._cxx is None:
            pytest.skip("morie._core not built")
    else:
        monkeypatch.setattr(native, "_cxx", None)
    return request.param


def _report(*paras):
    return "".join(f"<p>{p}</p>" for p in paras)


def test_incident_date_skips_the_notification_sentence(backend):
    # drid 670: the first dated sentence is the call to the SIU; the incident is the day before
    f = native.parse_report_html(
        _report(
            "The Investigation",
            "Notification of the SIU",
            "On July 9, 2019, at 1:00 a.m., the City of Kawartha Lakes Police Service contacted the SIU to "
            "report a serious injury. On July 8, 2019 at about 3:50 p.m., CKLPS were called to a residence.",
            "The Team",
        )
    )
    assert f["date_siu_notified_iso"] == "2019-07-09"
    assert f["date_of_incident_iso"] == "2019-07-08"


def test_incident_dated_relative_to_the_notification(backend):
    # drid 648: "two hours prior" puts the incident on the notification day, not an earlier event
    f = native.parse_report_html(
        _report(
            "The Investigation",
            "Notification of the SIU",
            "On September 12, 2019, at 3:00 p.m., the Peel Regional Police ( PRP ) notified the SIU of the "
            "serious injuries sustained by the Complainant during his arrest two hours prior.",
            "Incident Narrative",
            "On September 5, 2019, a home in Brampton was broken into.",
        )
    )
    assert f["date_siu_notified_iso"] == "2019-09-12"
    assert f["date_of_incident_iso"] == "2019-09-12"


def test_witness_rosters_in_role_sections(backend):
    # drid 4961: "Witness Officials ( WO )" with one numbered line per official
    f = native.parse_report_html(
        _report(
            "The Investigation",
            "Subject Official ( SO )",
            "SO Declined interview and to provide notes, as is the subject official's legal right",
            "Witness Officials ( WO )",
            *(f"WO #{i} Interviewed; notes received and reviewed" for i in range(1, 5)),
            "Civilian Witnesses ( CW )",
            *(f"CW #{i} Interviewed" for i in range(1, 4)),
            "Evidence",
        )
    )
    assert f["_language"] == "en"
    assert f["number_of_subject_officials"] == "1"
    assert f["number_of_witness_officials"] == "4"
    assert f["number_of_civilian_witnesses"] == "3"


def test_french_rosters(backend):
    # drid 4000: "AT n o 1" (a superscript o split off) under "Agents témoins"
    f = native.parse_report_html(
        _report(
            "L'enquête",
            "Agent impliqué",
            "AI n o 1 A refusé de participer à une entrevue",
            "Agents témoins",
            *(f"AT n o {i} A participé à une entrevue" for i in range(1, 5)),
            "Témoins civils",
            *(f"TC n o {i} A participé à une entrevue" for i in range(1, 8)),
            "Éléments de preuve",
        )
    )
    assert f["_language"] == "fr"
    assert f["number_of_subject_officials"] == "1"
    assert f["number_of_witness_officials"] == "4"
    assert f["number_of_civilian_witnesses"] == "7"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("SO #1 Interviewed. SO #2 Declined interview.", (2, "max ordinal SO #2")),
        ("WO #1 Interviewed. WO #2 Interviewed.", (0, "zero: witness officials only, no subject official named")),
        ("Les deux agents impliqués ont refusé.", (2, "plural cue 'deux agents impliqu'")),
        ("Nothing here.", (None, "UNRESOLVED: 'the SO'x0 'the subj off'x0")),
    ],
)
def test_resolver(backend, text, want):
    from morie.siu.corpus import resolve_subject_officials

    assert tuple(resolve_subject_officials(text)) == want


def test_compiled_parser_tolerates_a_field_clipped_inside_utf8():
    # the C++ clips specific_injuries by byte count and once ended a value inside
    # a multi-byte character; the binding raised UnicodeDecodeError on a real report
    if native._cxx is None:
        pytest.skip("morie._core not built")
    tail = "é" * 60  # 120 bytes: the 80-byte clip lands between the two bytes of an é
    html = _report("The Investigation", f"The man suffered a fractured left arm during the arrest {tail}.")
    f = native._cxx.siu_parse_report_html(html)
    assert f["specific_injuries"].startswith("fractured left arm")
    assert "�" in f["specific_injuries"] or f["specific_injuries"].endswith("é")
