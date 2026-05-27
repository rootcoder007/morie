# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Bootstrap CI."""

from morie.fn.boot import bootstrap_ci as _boot

bb8 = _boot
bootstrap_ci = _boot


def cheatsheet() -> str:
    return 'bb8() -> Bootstrap CI.'
