"""Tests for the native SIU core twin (morie.siu.native / llm / audit / CLI), cross-checked with the C++ core."""

import json

import pytest

pytest.importorskip("bs4")  # morie.siu imports the bs4-based scraper parser

from morie.siu import llm  # noqa: E402
from morie.siu.__main__ import main as siu_cli  # noqa: E402
from morie.siu.audit import siu_audit_panel  # noqa: E402
from morie.siu.corpus import PANEL_FIELDS  # noqa: E402
from morie.siu.native import html_to_text, parse_report_html, to_iso_date  # noqa: E402

FIXTURE = (
    "<p>SIU Director's Report - Case # 23-OFD-001</p><p>The Investigation</p><p>Notification of the SIU</p>"
    "<p>On January 6, 2023, the Barrie Police Service contacted the SIU with the following information. "
    "It reported that on January 5, 2023, in the City of Barrie, a 34-year-old man was seriously injured.</p>"
    "<p>The Team</p><p>Number of SIU Investigators assigned: 3</p>"
    "<p>Number of SIU Forensic Investigators assigned: 1</p>"
    "<p>Subject Officials</p><p>SO #1 Declined interview</p><p>SO #2 Interviewed</p>"
    "<p>Incident Narrative</p><p>The man sustained a fractured left arm during the arrest.</p>"
    "<p>Date: April 28, 2023</p><p>Electronically approved by</p><p>Joseph Martino</p><p>Director</p>"
)


def test_native_parser_fields():
    f = parse_report_html(FIXTURE)
    assert f["police_service"] == "Barrie Police Service"
    assert f["date_siu_notified_iso"] == "2023-01-06"
    assert f["date_of_director_decision_iso"] == "2023-04-28"
    assert f["location_of_call"] == "City of Barrie"
    assert (f["age_affected"], f["sex_gender_affected"]) == ("34", "man")
    assert f["number_of_subject_officials"] == "2"
    assert (f["siu_investigators"], f["siu_forensics_investigators"]) == ("3", "1")
    assert f["directors_name"] == "Joseph Martino"
    assert set(f) == {name for name, _, _ in PANEL_FIELDS} | {"_language"}
    assert to_iso_date("March 9 2021") == "2021-03-09"


def test_live_page_format():
    live = FIXTURE.replace("</p>", "</p>\r\n").replace(
        "<p>The Team</p>", "<p>Public Reports</p>\r\n<p>Director's Resource Committee</p><p>The Team</p>"
    )
    assert "\r" not in html_to_text(live)
    assert parse_report_html(live)["directors_name"] == "Joseph Martino"
    assert parse_report_html("<p>X Y</p><p>Jane Q. Doe<br>Interim Director</p>")["directors_name"] == "Jane Q. Doe"
    fr = "<p>Jane Q. Doe</p><p>Directrice</p>"
    assert parse_report_html(fr)["directors_name"] == "Jane Q. Doe"
    hd = "<p>Brampton Collision Between Police</p><p>The Peel Regional Police notified the SIU.</p>"
    assert parse_report_html(hd)["police_service"] == "Peel Regional Police"


def test_backend_resolution(monkeypatch):
    for v in (
        "MORIE_LLM_API",
        "MORIE_LLM_BASE",
        "MORIE_LLM_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_API_BASE_URL",
        "LLM_API_KEY",
        "OLLAMA_HOST",
        "OLLAMA_BASE_URL",
        "OLLAMA_API_KEY",
    ):
        monkeypatch.delenv(v, raising=False)
    assert llm.resolve().base == "http://localhost:11434"
    monkeypatch.setenv("MORIE_LLM_API", "openai")
    monkeypatch.setenv("LLM_API_BASE_URL", "http://vllm:8000")
    monkeypatch.setenv("LLM_API_KEY", "k3")
    b = llm.resolve()
    assert (b.api, b.base, b.key) == ("openai", "http://vllm:8000/v1", "k3")
    with pytest.raises(ValueError):
        llm.resolve(api="grpc")
    assert llm.list_models(chat=lambda m, p: "x") == []


def _fake(calls):
    def chat(model, prompt):
        calls.append(model)
        if prompt.startswith("Reply with the word OK."):
            return "" if model == "dead" else "OK"
        if "You are the AUDITOR" in prompt:
            return '<think>..</think>{"police_service": "Barrie Police Service", "number_of_subject_officials": 2}'
        return '{"police_service": {"value": "Barrie", "quote": "Barrie", "confidence": "high"}}'

    return chat


def test_audit_panel_modes_and_chain():
    calls = []
    r1 = siu_audit_panel("report", {}, mode=1, readers=["dead", "r1"], chat=_fake(calls))
    assert r1["json"] == '{"police_service":"Barrie"}' and calls.count("r1") == 2
    calls.clear()
    r4 = siu_audit_panel("report", {}, mode=4, readers=["r1", "r2"], auditors=["a1", "a2"], chat=_fake(calls))
    assert r4["json"] == '{"number_of_subject_officials":2,"police_service":"Barrie Police Service"}'
    assert sum(c in ("r1", "r2") for c in calls) == 3 + 2 and sum(c in ("a1", "a2") for c in calls) == 2 + 2
    with pytest.raises(RuntimeError, match="healthy"):
        siu_audit_panel("report", {}, readers=["dead"], chat=_fake([]))
    per = siu_audit_panel("report", {}, mode=2, readers=["r1"], reader_granularity="per-field", chat=_fake([]))
    assert json.loads(per["json"])["police_service"] == "Barrie Police Service"


def test_cli(tmp_path, capsys):
    h = tmp_path / "r.html"
    h.write_text(FIXTURE, encoding="utf-8")
    t = tmp_path / "r.txt"
    t.write_text("Subject Officials\nSO #1 Interviewed\n", encoding="utf-8")
    assert siu_cli(["version"]) == 0
    assert siu_cli(["siu", "parse", str(h)]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["directors_name"] == "Joseph Martino"
    assert siu_cli(["resolve", str(t)]) == 0
    assert "subject_officers=1" in capsys.readouterr().out
    assert siu_cli(["bogus"]) == 2


def test_sections_are_read_from_the_body_not_the_table_of_contents():
    """Report pages open with a contents list repeating every section title; the parser read that."""
    from morie.siu import native

    html = (
        "<ul><li><a>The Investigation</a></li><li><a>Incident Narrative</a></li><li><a>Evidence</a></li></ul>"
        "<h2>The Investigation</h2><p>At approximately 11:46 a.m. on August 3rd, 2017, the Guelph Police "
        "Service (GPS) notified the SIU of an injury.</p>"
        "<h2>Incident Narrative</h2><p>Just prior to 10:00 a.m. on August 2nd, 2017, three men attempted a robbery.</p>"
        "<h2>Evidence</h2><p>none</p>"
    )
    r = native.parse_report_html(html)
    assert r["police_service"] == "Guelph Police Service"
    assert r["date_siu_notified_iso"] == "2017-08-03"
    assert r["date_of_incident_iso"] == "2017-08-02"


def test_dates_in_ordinal_french_and_abbreviated_forms():
    from morie.siu import native

    got = [
        native.to_iso_date(s)
        for s in (
            "3 août 2017",
            "August 3rd, 2017",
            "1er janvier 2020",
            "December 1st, 2020",
            "Aug 3, 2017",
            "3 August 2017",
            "Sept. 5, 2019",
            "soon",
        )
    ]
    assert got == ["2017-08-03", "2017-08-03", "2020-01-01", "2020-12-01", "2017-08-03", "2017-08-03", "2019-09-05", ""]
    assert (
        native.html_to_text("<p>Fran&ccedil;ais &eacute;t&eacute; &#233; &hellip;</p>").strip() == "Français été é ..."
    )


def _page(*paras):
    return "<html><body>" + "".join(f"<p>{p}</p>" for p in paras) + "</body></html>"


@pytest.mark.parametrize(
    ("paras", "want"),
    [
        # the subject official's service, not the force that notified the SIU
        (
            (
                "The Lakeshore Police Service ( LPS ) notified the SIU of the injury.",
                "Analysis and Director's Decision",
                "The Complainant was hurt while in the custody of the LPS.",
                "The SO of the Hillcrest Police Service was identified as the subject official.",
            ),
            "Hillcrest Police Service",
        ),
        # the sentence before a subject-official sentence that names no service
        (
            (
                "The Lakeshore Police Service notified the SIU.",
                "Analysis and Director's Decision",
                "The Complainant was arrested by Riverton Police Service officers.",
                "The SO was identified as the subject official.",
            ),
            "Riverton Police Service",
        ),
        # neither: the service the analysis names most
        (
            (
                "The Lakeshore Police Service notified the SIU.",
                "Analysis and Director's Decision",
                "The Complainant fell from a balcony.",
                "The SO was identified as the subject official.",
                "Riverton Police Service records were reviewed.",
                "Riverton Police Service officers had attended; the Lakeshore Police Service assisted.",
            ),
            "Riverton Police Service",
        ),
        # the case-number letter: P is the Ontario Provincial Police, O any other service
        (
            (
                "Director's Report for Case # 21-PCI-500",
                "Analysis and Director's Decision",
                "The SO of the Hillcrest Police Service assisted the OPP in the arrest.",
            ),
            "Ontario Provincial Police",
        ),
        (
            (
                "Director's Report for Case # 21-OCI-502",
                "Analysis and Director's Decision",
                "The SO had assisted the OPP before Riverton Police Service officers made the arrest.",
            ),
            "Riverton Police Service",
        ),
        # French: the agent impliqué's service
        (
            (
                "Témoins civils",
                "Agents impliqués",
                "Notification de l’UES",
                "La Police provinciale de l’Ontario ( PPO ) a avisé l’UES de la blessure.",
                "Analyse et décision du directeur",
                "Un agent du Service de police de Rivièreville, l’AI , a été désigné comme agent impliqué.",
            ),
            "Service de police de Rivièreville",
        ),
        # a legacy report's Police service header, alone or as part of a full name
        (
            (
                "File #: 10-OFD-900 Police service: Lakeshore Incident date: May 9, 2010",
                "Notification of the SIU The Riverton Police Service notified the SIU.",
                "Officers of the Lakeshore Police Service attended.",
            ),
            "Lakeshore Police Service",
        ),
        (
            (
                "File #: 10-OFD-901 Police service: Elmwood Incident date: May 9, 2010",
                "Notification of the SIU The Riverton Police Service notified the SIU.",
            ),
            "Elmwood",
        ),
    ],
)
def test_police_service_is_the_subject_officials_service(paras, want):
    assert parse_report_html(_page(*paras))["police_service"] == want
