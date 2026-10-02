"""`morie edit` opens $EDITOR (arguments allowed) on the file and runs it with --run."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from morie.editor import edit_file

STUB = "import sys\nopen(sys.argv[1], 'a').write('print(6 * 7)\\n')\n"


def test_edit_file_honours_editor_with_arguments(tmp_path, monkeypatch):
    stub = tmp_path / "stub_editor.py"
    stub.write_text(STUB)
    monkeypatch.setenv("EDITOR", f'"{sys.executable}" "{stub}"')
    monkeypatch.delenv("VISUAL", raising=False)
    target = tmp_path / "f.py"
    target.write_text("")
    assert edit_file(str(target)) == 0
    assert "print(6 * 7)" in target.read_text()


def test_edit_without_editor_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setattr("morie.editor.shutil.which", lambda name: None)
    assert edit_file(str(tmp_path / "f.py")) == 1
    assert "EDITOR" in capsys.readouterr().err


def test_edit_then_run(tmp_path):
    pytest.importorskip("morie._exec_guard")  # exec is source-tree only
    stub = tmp_path / "stub_editor.py"
    stub.write_text(STUB)
    env = dict(os.environ, EDITOR=f'"{sys.executable}" "{stub}"', MORIE_NO_UPDATE_CHECK="1")
    env.pop("VISUAL", None)
    env.pop("MORIE_NO_EXEC", None)
    r = subprocess.run(
        [sys.executable, "-m", "morie.runner", "edit", str(tmp_path / "f.py"), "--run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert "42" in r.stdout
