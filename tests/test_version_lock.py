# SPDX-License-Identifier: AGPL-3.0-or-later
"""The version is declared in four places; they must agree.

morie ships a Python distribution, a citation record, and a vendored R
arm, and it is released in lockstep with rmorie. Nothing enforced that,
so CITATION.cff sat at 1.2.2 while pyproject.toml and the R DESCRIPTION
had moved to 1.2.3 -- a citation that points at a version which never
contained the work being cited.

These assertions read the files, so they fail on the next drift rather
than after someone notices.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    txt = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # the [project] version, not a dependency pin
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', txt)
    assert m, "no version in pyproject.toml"
    return m.group(1)


def _citation_version() -> str | None:
    p = ROOT / "CITATION.cff"
    if not p.exists():
        return None
    m = re.search(r'(?m)^version:\s*"?([^"\s]+)"?', p.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _r_description_version() -> str | None:
    p = ROOT / "r-package" / "morie" / "DESCRIPTION"
    if not p.exists():
        return None
    m = re.search(r'(?m)^Version:\s*(\S+)', p.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _version_file() -> str | None:
    p = ROOT / "VERSION"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def test_the_version_file_is_the_source_of_truth():
    # scripts/version-inventory.sh computes CURRENT vs STALE against the
    # root VERSION file, so a VERSION that lags the release manifests
    # marks every correct file stale and fails the drift gate. That is
    # what happened: VERSION sat at 1.2.2 while pyproject.toml and the R
    # DESCRIPTION were 1.2.3.
    vf = _version_file()
    if vf is None:
        pytest.skip("no VERSION file in this tree")
    assert vf == _pyproject_version(), (
        "VERSION says %s, pyproject.toml says %s -- VERSION is what the "
        "drift gate compares against" % (vf, _pyproject_version()))


def test_version_is_a_release_number():
    v = _pyproject_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), v


def test_citation_matches_the_distribution():
    cff = _citation_version()
    if cff is None:
        pytest.skip("no CITATION.cff in this tree")
    assert cff == _pyproject_version(), (
        "CITATION.cff says %s, pyproject.toml says %s -- a citation must "
        "point at a version that contains the work" % (cff, _pyproject_version()))


def test_the_vendored_r_arm_matches_the_python_arm():
    rv = _r_description_version()
    if rv is None:
        pytest.skip("no vendored r-package in this tree")
    assert rv == _pyproject_version(), (
        "r-package/morie/DESCRIPTION says %s, pyproject.toml says %s -- the "
        "two arms of the same release must not diverge"
        % (rv, _pyproject_version()))


def test_installed_metadata_agrees_when_installed_from_this_tree():
    # A stale editable/site-packages install is the usual reason a test
    # run disagrees with the source tree, so say which it is rather than
    # failing obscurely later.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version
    try:
        installed = pkg_version("morie")
    except PackageNotFoundError:
        pytest.skip("morie is not installed")
    if installed != _pyproject_version():
        pytest.skip(
            "installed morie is %s but this tree is %s: reinstall to test the "
            "tree (pip install -e .)" % (installed, _pyproject_version()))
    import morie
    assert morie.__version__ == installed
