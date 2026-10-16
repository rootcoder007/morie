"""The R bridge's native adapters give the numbers of the reference packages they replace.

Each ``tests/r_bridge_parity/<group>_adapters.R`` holds, per ``r_*`` command, the reference call
(car, lme4, survey, ...), a comparison and example data; ``run_parity.R`` runs the adapter that
ships (``src/morie/rscripts/bridge_natives.R``) against it. The reference packages are test-only:
morie never calls them. A command whose reference package is not installed is skipped.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent / "r_bridge_parity"
NATIVES = Path(__file__).parents[1] / "src" / "morie" / "rscripts" / "bridge_natives.R"
GROUPS = sorted(HERE.glob("*_adapters.R"))


def _morie_r_package():
    if shutil.which("Rscript") is None:
        return None
    probe = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            'for (p in c("morie", "rmorie")) if (requireNamespace(p, quietly = TRUE)) { cat(p); break }',
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return (probe.stdout.strip() or None) if probe.returncode == 0 else None


PKG = _morie_r_package()


def test_the_shipped_natives_are_generated_from_these_adapters():
    names = set()
    for g in GROUPS:
        names |= {line.split('"')[1] for line in g.read_text().splitlines() if line.startswith('ADAPTERS[["')}
    shipped = {line.split('"')[1] for line in NATIVES.read_text().splitlines() if line.startswith('NATIVE[["')}
    assert shipped and shipped <= names, shipped - names


@pytest.mark.skipif(PKG is None, reason="Rscript or morie's R package not installed")
@pytest.mark.parametrize("group", GROUPS, ids=[g.stem for g in GROUPS])
def test_native_adapters_match_their_reference(group):
    out = subprocess.run(
        ["Rscript", str(HERE / "run_parity.R"), str(group), str(NATIVES), PKG],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    rows = [line.split(" ", 3) for line in out.stdout.splitlines() if line.strip()]
    assert rows, out.stderr[-2000:]
    bad = [" ".join(r) for r in rows if r[1] in ("MISMATCH", "ERROR")]
    assert bad == [], bad
