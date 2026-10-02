# SPDX-License-Identifier: AGPL-3.0-or-later
"""R output without a trailing newline (cat(42)) must survive the sentinel read; Python values bridge into R."""

import shutil

import pytest

from morie.polyglot import PolyglotEngine

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")


def test_r_output_without_newline_is_kept():
    e = PolyglotEngine(polyglot=True, auto_detect=True)
    r = e.execute("R> cat(6 * 7)")
    assert r.success and r.stdout.strip() == "42"
    r = e.execute("R> cat('a\\n'); cat('b')")
    assert r.stdout.splitlines() == ["a", "b"]


def test_python_value_bridges_into_r():
    e = PolyglotEngine(polyglot=True, auto_detect=True)
    assert e.execute("x = 20").success
    r = e.execute("R> cat(x * 2 + 2)")
    assert r.stdout.strip() == "42"
