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
    assert "pak::pkg_install(c('rmoriebricklayer', 'rmoriedata', 'rmorie'))" in runiv and bricklayer.RUNIV in runiv
    assert "install.packages(c('rmoriebricklayer','rmoriedata','rmorie')" in runiv  # the fallback
    gh = bricklayer._r_install_expr(github=True)
    from morie import __version__ as v

    # pinned to the release tag, like the r-universe route; pak first, remotes as the fallback
    assert f"'rootcoder007/morie/r-package/morie@v{v}'" in gh
    assert f"remotes::install_github('rootcoder007/morie@v{v}', subdir = 'r-package/morie'" in gh
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
    assert "remotes::install_github('rootcoder007/morie@v" in out
    rc = bricklayer.run(p.parse_args(["r-install"]))
    out = capsys.readouterr().out
    assert rc == 0 and "'rmorie'))" in out


def test_r_install_expr_sets_repos_and_upgrades_companions():
    # under Rscript a bare install.packages() stops with "trying to use CRAN without
    # setting a mirror"; an old companion already installed must be replaced
    import re

    for github in (False, True):
        expr = bricklayer._r_install_expr(github=github)
        # every call (not the "install.packages()" named in the fallback message)
        calls = re.findall(r"install\.packages\((?!\))([^;]*)", expr)
        assert calls and all("repos" in c for c in calls), expr
        assert "upgrade = 'never'" not in expr and "upgrade = 'always'" in expr
        assert "pak::pkg_install(" in expr and "falling back to install.packages()" in expr
        assert "'rmoriebricklayer'" in expr and "'rmoriedata'" in expr
        # r-universe ahead of CRAN wherever both are listed
        assert f"c('{bricklayer.RUNIV}','{bricklayer.CRAN}')" in expr
        assert f"c('{bricklayer.CRAN}','{bricklayer.RUNIV}')" not in expr


def test_install_sh_sets_repos_and_upgrades():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("install.sh", "docs/source/_extra/install.sh"):
        text = (root / rel).read_text()
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if "install.packages(" in code and "install.packages()" not in code:
                assert "repos" in code, (rel, line)
        assert 'upgrade = "never"' not in text, rel
        assert "pak::pkg_install(" in text, rel
