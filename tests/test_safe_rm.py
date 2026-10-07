"""Recursive deletes refuse anything morie did not make (a bad cleanup once removed a checkout)."""

from __future__ import annotations

import os

import pytest

from morie._safe_rm import refuse_reason, rmtree_owned


def test_protected_directories_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert refuse_reason("/") == "it is the filesystem root"
    assert refuse_reason(os.path.expanduser("~")) == "it is the home directory"
    assert refuse_reason(".") == "it is the working directory or one of its parents"
    assert refuse_reason(tmp_path.parent) == "it is the working directory or one of its parents"
    repo = tmp_path / "checkout"
    (repo / ".git").mkdir(parents=True)
    assert refuse_reason(repo) == "it is a git checkout"
    with pytest.raises(PermissionError, match="git checkout"):
        rmtree_owned(repo, owned=True)
    assert repo.exists()


def test_only_an_owned_directory_is_removed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "scratch"
    (d / "sub").mkdir(parents=True)
    with pytest.raises(PermissionError, match="not created by morie"):
        rmtree_owned(d, owned=False)
    assert d.exists()
    rmtree_owned(d, owned=True)
    assert not d.exists()


def test_interactive_remove_needs_its_own_marker(tmp_path, monkeypatch):
    from morie import _interactive

    target = tmp_path / "somebody-elses-dir"
    target.mkdir()
    (target / "work.txt").write_text("keep me")
    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(target))
    lines = []
    assert _interactive.remove(out=lines.append) == 1
    assert (target / "work.txt").read_text() == "keep me"
    assert "no VERSION file" in lines[0]
    (target / "VERSION").write_text("1.4.0\n")
    assert _interactive.remove(out=lines.append) == 0
    assert not target.exists()
