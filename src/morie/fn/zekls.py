"""Circular scan statistic: Kulldorff's circular scan (implementation in :mod:`morie.fn.scanstat`)."""

from .scanstat import kulldorff_scan as scan_circular

scan = scan_circular


def cheatsheet() -> str:
    return "scan_circular(coords, cases, pop) -> circular spatial scan statistic (Kulldorff 1997)."


# compact alias per ledger/NAMING.md
scancircular = scan_circular
