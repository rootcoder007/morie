"""The trivial-assertion harvest must stay in step with the detector it replays.

``tests/fn/_audit/harvest_trivial.py`` reproduces the ``FALSE-POSITIVE RISK``
fixture in ``tests/conftest.py`` statically, so the 3,621 flagged tests can be
listed and tracked without a 12-minute pytest run. That only holds while the
two use the same pattern set -- if the fixture gains a pattern and the harvest
does not, ``trivial_tests.csv`` silently starts counting something else.

This pins them together. It is deliberately fast: the full harvest (which
walks 36k test files and skeletonises every function in ``morie.fn``) runs in
CI via ``harvest_trivial.py --check``, not here.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARVEST = REPO / "tests" / "fn" / "_audit" / "harvest_trivial.py"
CSV_PATH = REPO / "tests" / "fn" / "_audit" / "trivial_tests.csv"


def _load_harvest():
    spec = importlib.util.spec_from_file_location("harvest_trivial", HARVEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_conftest():
    """Load tests/conftest.py by path.

    `import tests.conftest` only resolves when the repository root is on
    sys.path, which is true when pytest is run from the checkout and not
    when it is run against an installed package -- that is why this
    passed locally and failed in CI with "No module named 'tests'".
    Loading by path works either way, and mirrors _load_harvest above.
    """
    spec = importlib.util.spec_from_file_location(
        "morie_tests_conftest", REPO / "tests" / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pattern_set_matches_the_conftest_fixture():
    """The harvest's copy of _TRIVIAL_PATTERNS is identical to conftest's."""
    tests_conftest = _load_conftest()

    harvest = _load_harvest()
    assert harvest.TRIVIAL_PATTERNS == tests_conftest._TRIVIAL_PATTERNS, (
        "harvest_trivial.TRIVIAL_PATTERNS has drifted from "
        "tests/conftest.py::_TRIVIAL_PATTERNS. Update both in the same commit, "
        "then regenerate trivial_tests.csv."
    )


def test_trivial_detection_agrees_with_the_fixture_condition():
    """Spot-check the replayed logic on the shapes that actually occur.

    The three patterns below cover 3,607 of the 3,621 flagged tests; the
    negative cases are the ones a wrong prefix match would misclassify.
    """
    harvest = _load_harvest()
    trivial = lambda src: harvest._is_trivial(harvest._assert_lines(src))  # noqa: E731

    assert trivial("def t():\n    r = f()\n    assert r.value is not None\n")
    assert trivial("def t():\n    r = f()\n    assert r.name\n")
    assert trivial("def t():\n    result = f()\n    assert result is not None\n")

    # A domain check alongside a trivial one is NOT flagged -- the fixture
    # requires every assert to be trivial.
    assert not trivial("def t():\n    assert r.value is not None\n    assert r.value == 3\n")
    # No asserts at all is not flagged either.
    assert not trivial("def t():\n    f()\n")
    # `assert r.names` starts with `assert r.name`, so a naive prefix match
    # would flag it. It is genuinely flagged by the fixture too -- this pins
    # that the harvest inherits the quirk rather than silently diverging.
    assert trivial("def t():\n    assert r.names == ['a']\n")


def test_csv_is_present_and_well_formed():
    """The committed harvest is the durable record; keep it readable."""
    assert CSV_PATH.exists(), f"{CSV_PATH} missing -- run harvest_trivial.py"
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "trivial_tests.csv is empty"
    assert {"nodeid", "target_kind", "skeleton_siblings"} <= set(rows[0])
    assert all(r["target_kind"] in {"real", "stub", "unknown"} for r in rows)
