# SPDX-License-Identifier: AGPL-3.0-or-later
"""`morie interactive`: the source-tree-only modules can be added to an installed copy, verified, and removed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from morie import _interactive as inter

REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "morie"
pytestmark = pytest.mark.skipif(
    not all((REPO_SRC / n).is_file() for n in inter.FILES), reason="needs the source checkout's src/morie"
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_manifest_matches_the_source_tree():
    """The manifest the wheel ships must describe the files at the same commit."""
    manifest = json.loads((REPO_SRC / "_interactive_manifest.json").read_text(encoding="utf-8"))["files"]
    assert set(manifest) == set(inter.FILES)
    for name in inter.FILES:
        assert manifest[name] == _sha(REPO_SRC / name), f"{name}: run scripts/interactive_manifest.py"


def test_hash_ignores_crlf_line_endings(tmp_path):
    """A Windows checkout (autocrlf) must verify against the manifest built from LF files."""
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\nprint(x)\n")
    crlf.write_bytes(b"x = 1\r\nprint(x)\r\n")
    assert inter.sha256_of(lf) == inter.sha256_of(crlf)
    assert inter.sha256_of(lf) == hashlib.sha256(b"x = 1\nprint(x)\n").hexdigest()


def test_default_ref_is_the_release_tag_or_main():
    assert inter.default_ref("1.3.9") == "v1.3.9"
    assert inter.default_ref("1.3.9.dev3") == "main"
    assert inter.default_ref("0.0.0+unknown") == "main"
    assert inter.default_ref("") == "main"


def test_data_dir_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(tmp_path / "layer"))
    assert inter.data_dir() == tmp_path / "layer"


def test_install_from_a_local_directory_then_activate_and_remove(tmp_path, monkeypatch):
    target = tmp_path / "layer"
    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(target))
    monkeypatch.setattr(inter, "package_version", lambda: "9.9.9")
    lines: list[str] = []
    assert inter.install(source=str(REPO_SRC), verify=True, out=lines.append) == 0
    assert (target / "VERSION").read_text(encoding="utf-8").strip() == "9.9.9"
    for name in inter.FILES:
        assert _sha(target / name) == _sha(REPO_SRC / name)
    assert any("verified against the bundled manifest" in ln for ln in lines)

    # `import morie` joins the directory only for the matching version
    path: list[str] = []
    assert inter.activate(path, "9.9.9") is True and path == [str(target)]
    assert inter.activate(path, "9.9.9") is True and path == [str(target)]  # idempotent
    other: list[str] = []
    assert inter.activate(other, "1.0.0") is False and other == []

    out: list[str] = []
    assert inter.status(out=out.append) == 0
    assert any("active for morie 9.9.9" in ln or "source checkout" in ln for ln in out)

    assert inter.remove(out=out.append) == 0
    assert not target.exists()
    assert inter.remove(out=out.append) == 0  # a second remove is a no-op, not an error


def test_tampered_file_is_refused_unless_verification_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(tmp_path / "layer"))
    monkeypatch.setattr(inter, "package_version", lambda: "9.9.9")
    src = tmp_path / "src"
    src.mkdir()
    for name in inter.FILES:
        (src / name).write_bytes((REPO_SRC / name).read_bytes())
    (src / "agent.py").write_text("# tampered\n", encoding="utf-8")
    lines: list[str] = []
    assert inter.install(source=str(src), verify=True, out=lines.append) == 1
    assert any("agent.py" in ln and "differ" in ln for ln in lines)
    assert not (tmp_path / "layer" / "VERSION").exists()
    assert inter.install(source=str(src), verify=False, out=lines.append) == 0
    assert (tmp_path / "layer" / "agent.py").read_text(encoding="utf-8") == "# tampered\n"


def test_missing_manifest_blocks_a_verified_install(tmp_path, monkeypatch):
    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(tmp_path / "layer"))
    monkeypatch.setattr(inter, "manifest", lambda: {})
    lines: list[str] = []
    assert inter.install(source=str(REPO_SRC), verify=True, out=lines.append) == 1
    assert any("no manifest" in ln for ln in lines)


def test_github_route_builds_the_tagged_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(tmp_path / "layer"))
    monkeypatch.setattr(inter, "package_version", lambda: "1.3.9")
    seen: list[str] = []

    def fake_download(url: str, dest: Path, label: str) -> None:
        seen.append(url)
        dest.write_bytes((REPO_SRC / dest.name).read_bytes())

    monkeypatch.setattr(inter, "_download", fake_download)
    assert inter.install(out=lambda s: None) == 0
    assert seen == [inter.RAW_URL.format(ref="v1.3.9", name=n) for n in inter.FILES]
    assert (tmp_path / "layer" / "VERSION").read_text(encoding="utf-8").strip() == "1.3.9"


def test_a_missing_ref_is_reported_in_words(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(tmp_path / "layer"))

    def gone(url: str, dest: Path, label: str) -> None:
        raise HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(inter, "_download", gone)
    lines: list[str] = []
    assert inter.install(ref="no-such-ref", out=lines.append) == 1
    assert any("no-such-ref" in ln and "--ref" in ln for ln in lines)


def test_cli_verbs_route_to_the_module(tmp_path, monkeypatch, capsys):
    import sys

    from morie import runner

    monkeypatch.setenv("MORIE_INTERACTIVE_DIR", str(tmp_path / "layer"))

    def run(*argv: str) -> int:
        monkeypatch.setattr(sys, "argv", ["morie", *argv])
        return runner.main()

    assert run("interactive", "status") == 0
    assert "interactive layer directory" in capsys.readouterr().out
    assert run("interactive", "install", "--from", str(REPO_SRC), "--no-verify") == 0
    assert "Installed the interactive layer" in capsys.readouterr().out
    assert run("interactive", "remove") == 0
    assert "Removed" in capsys.readouterr().out


def test_offer_is_silent_off_a_terminal(monkeypatch, capsys):
    """Pipes and CI get the one-line instruction and no stdin read."""
    import sys

    from morie import runner

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("stdin must not be read off a terminal"))
    assert runner._add_interactive_layer("The agent") is False
    out = capsys.readouterr().out
    assert "not bundled in this install" in out and "morie interactive install" in out


def test_offer_installs_on_yes(monkeypatch, capsys):
    import sys

    from morie import runner

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("MORIE_NO_PROMPT", raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    calls: list[str] = []
    monkeypatch.setattr(inter, "install", lambda **kw: calls.append("install") or 0)
    monkeypatch.setattr(inter, "activate", lambda path, version: calls.append("activate") or True)
    assert runner._add_interactive_layer("The agent") is True
    assert calls == ["install", "activate"]
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert runner._add_interactive_layer("The agent") is False
    monkeypatch.setenv("MORIE_NO_PROMPT", "1")
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("MORIE_NO_PROMPT must suppress the question"))
    assert runner._add_interactive_layer("The agent") is False
