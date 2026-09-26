"""Tests for morie._parity — verify migration parity is complete."""

import pytest

from morie._parity import MORIE_SCRIPT_MAP, build_parity_matrix, summarize_parity_matrix


@pytest.fixture
def _RTESTS_DIR(tmp_path):
    """A minimal legacy epiml tree (the real one is not part of this repo):
    one mapped analysis script and a NAMESPACE whose export the current R
    package already provides."""
    legacy = tmp_path / "epiml"
    legacy.mkdir()
    (legacy / next(iter(MORIE_SCRIPT_MAP))).write_text("# legacy script\n")
    ns = legacy / "packages" / "epiml" / "NAMESPACE"
    ns.parent.mkdir(parents=True)
    ns.write_text("export(epiml_paths)\nexport(epiml_gone)\n")
    repo = tmp_path / "repo"
    cur = repo / "r-package" / "morie" / "NAMESPACE"
    cur.parent.mkdir(parents=True)
    cur.write_text("export(morie_paths)\n")
    return legacy, repo


def test_parity_matrix_has_expected_kinds(_RTESTS_DIR):
    matrix = build_parity_matrix(*_RTESTS_DIR)
    assert "kind" in matrix.columns
    assert len(matrix) > 0
    assert "analysis_module" in matrix["kind"].values


def test_parity_summary_counts_rows(_RTESTS_DIR):
    matrix = build_parity_matrix(*_RTESTS_DIR)
    summary = summarize_parity_matrix(matrix)
    assert summary.total_rows == len(matrix)
    assert summary.already_present >= 1
