# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the stdlib retry decorator that replaced stamina in siu/_scraper."""

import pytest

from morie.siu._scraper import _retry


class Boom(Exception):
    pass


class Other(Exception):
    pass


def test_returns_on_first_success():
    calls = []

    @_retry(on=(Boom,), attempts=3, wait_initial=0, wait_max=0, wait_jitter=0)
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    calls = []

    @_retry(on=(Boom,), attempts=3, wait_initial=0, wait_max=0, wait_jitter=0)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise Boom("transient")
        return "recovered"

    assert f() == "recovered"
    assert len(calls) == 3


def test_reraises_after_exhausting_attempts():
    calls = []

    @_retry(on=(Boom,), attempts=3, wait_initial=0, wait_max=0, wait_jitter=0)
    def f():
        calls.append(1)
        raise Boom("always")

    with pytest.raises(Boom):
        f()
    assert len(calls) == 3  # tried exactly `attempts` times, no more


def test_does_not_retry_unlisted_exception():
    calls = []

    @_retry(on=(Boom,), attempts=3, wait_initial=0, wait_max=0, wait_jitter=0)
    def f():
        calls.append(1)
        raise Other("not retried")

    with pytest.raises(Other):
        f()
    assert len(calls) == 1  # unlisted exception propagates immediately
