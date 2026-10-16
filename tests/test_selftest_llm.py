# SPDX-License-Identifier: AGPL-3.0-or-later
"""selftest's LLM check accepts every route detect_available_provider() can return."""

import pytest

import morie.llm
from morie import selftest
from morie.providers import IDS


@pytest.mark.parametrize("route", ["own", "ollama", "local", *IDS])
def test_llm_detection_accepts_every_route(monkeypatch, route):
    monkeypatch.setattr(morie.llm, "detect_available_provider", lambda: route)
    assert selftest._test_llm_detection() == f"LLM provider: {route}"


def test_llm_detection_refuses_retired_hosted(monkeypatch):
    monkeypatch.setattr(morie.llm, "detect_available_provider", lambda: "hosted")
    with pytest.raises(RuntimeError, match="Unknown provider: hosted"):
        selftest._test_llm_detection()
