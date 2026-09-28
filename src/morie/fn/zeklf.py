"""Kulldorff spatial scan statistic (the implementation lives in :mod:`morie.fn.scanstat`)."""

from .scanstat import kulldorff_scan

kull = kulldorff_scan


def cheatsheet() -> str:
    return "kulldorff_scan(coords, cases, pop) -> Kulldorff circular spatial scan statistic."


# compact alias per ledger/NAMING.md
kulldorffscan = kulldorff_scan
