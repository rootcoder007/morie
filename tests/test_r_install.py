# SPDX-License-Identifier: AGPL-3.0-or-later
"""`morie r-install`: the R-side installer, from r-universe or from GitHub."""

from morie import bricklayer
from morie.runner import build_parser


def test_r_install_parses_like_bricklayer():
    p = build_parser()
    a = p.parse_args(["r-install", "--github", "--check"])
    assert a.command == "r-install" and a.github and a.check
    b = p.parse_args(["bricklayer", "--yes"])
    assert b.command == "bricklayer" and b.yes and not b.github


def test_r_install_expr_names_both_routes():
    runiv = bricklayer._r_install_expr(github=False)
    assert "install.packages('rmorie'" in runiv and bricklayer.RUNIV in runiv
    gh = bricklayer._r_install_expr(github=True)
    assert "remotes::install_github('rootcoder007/morie', subdir = 'r-package/morie')" in gh
    assert "remotes" in gh and "install.packages('remotes'" in gh


def test_r_install_without_r_prints_the_command(monkeypatch, capsys):
    monkeypatch.setattr(bricklayer, "_rscript", lambda: None)
    monkeypatch.setattr(bricklayer, "_have_r_morie", lambda: False)
    monkeypatch.setattr(bricklayer, "_py_backend_ok", lambda: True)
    p = build_parser()
    rc = bricklayer.run(p.parse_args(["r-install", "--github"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "R is not installed" in out
    assert "remotes::install_github('rootcoder007/morie'" in out
    rc = bricklayer.run(p.parse_args(["r-install"]))
    out = capsys.readouterr().out
    assert rc == 0 and "install.packages('rmorie'" in out
