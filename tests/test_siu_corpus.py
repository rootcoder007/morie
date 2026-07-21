# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-wrong rule engine + corpus-first surface.

Every regression class from the 2,182-report zero-wrong pass, mirrored from
rmoriebricklayer's test-siu-parse.R so the Python port can never drift from
the canonical C++ rules. Fully offline.
"""
import pytest

from morie.siu.corpus import (
    _panel_extract,
    resolve_subject_officials,
    siu_resolve_so,
    strip_boilerplate,
)


def count(text):
    return resolve_subject_officials(text)[0]


def test_roster_ordinals():
    assert count("SO #1 Interviewed\nSO #2 Declined\nSO #3 Interviewed") == 3


def test_singular_and_glossary_negation():
    assert count("The SO was interviewed. WO #1 is not a subject official.") == 1


def test_direct_zero_assertion():
    assert count("No subject official was designated.") == 0


def test_unresolved_never_guesses():
    n, reason = resolve_subject_officials("The report discusses the incident only.")
    assert n is None
    assert "UNRESOLVED" in reason


def test_glossary_note_never_feeds_zero_both_eras():
    g1 = ("A witness officer is a police officer who, in the opinion of "
          "the SIU Director, is involved in the incident under "
          "investigation but is not a subject officer. "
          "The SO was interviewed.")
    assert count(g1) == 1
    g2 = ("An official who, in the SIU Director’s opinion, is "
          "involved in the incident under investigation but is not a "
          "subject official. The SO declined an interview.")
    assert count(g2) == 1


def test_ordinals_case_strict_hash_required_roster_anchored():
    # 'also 59' must not read as SO 59
    assert count("The pursuit also 59 seconds later ended. The SO stopped the car.") == 1
    # lone narrative ordinal without the #1 anchor is another force's shorthand
    assert count("WO #6 and SO #7 of the neighbouring service stopped a Jeep. "
                 "The SO was interviewed.") == 1
    # a real roster anchors at #1
    assert count("SO #1 and SO #2 were interviewed.") == 2


def test_team_section_beats_narrative_and_tolerates_mislabelled_roster():
    txt = ("Subject Officers SO #1 Declined interview. "
           "SO #1 Declined interview, notes received.\n"
           "Incident Narrative\nOfficers responded.")
    assert count(txt) == 2  # 2 status entries beat the repeated ordinal


def test_nbsp_does_not_blind_the_scanners():
    txt = ("Subject Officers SO #1 Declined interview. "
           "SO #2 Declined interview.")
    assert count(txt) == 2


def test_plural_cue():
    assert count("The two subject officials responded to the call.") == 2


def test_privacy_paragraph_never_counts():
    txt = ("This information may include the Subject Officer name(s) "
           "and other evidence. No subject official was designated.")
    assert count(txt) == 0


def test_strip_boilerplate_normalises_nbsp():
    assert " " not in strip_boilerplate("SO #1")


def test_resolve_so_argument_contract():
    with pytest.raises(ValueError):
        siu_resolve_so()
    with pytest.raises(ValueError):
        siu_resolve_so(text="x", drid=1)


def test_panel_extract_tolerates_prose_and_variants():
    raw = ('Sure! Here is the JSON:\n{"police_service": {"value": '
           '"Peel Regional Police", "quote": "the Peel..."}, '
           '"number_of_subject_officers": {"value": 2, "quote": "SO #2"}}')
    out = _panel_extract(raw, ["police_service", "number_of_subject_officers",
                               "directors_name"])
    assert out["police_service"] == "Peel Regional Police"
    assert out["number_of_subject_officers"] == "2"
    assert out["directors_name"] is None


def test_panel_extract_per_field_dict():
    raw = {"directors_name": '{"directors_name": {"value": "Joseph Martino"}}'}
    out = _panel_extract(raw, ["directors_name"])
    assert out["directors_name"] == "Joseph Martino"
