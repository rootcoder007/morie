"""tsscr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsscr import tsscr


def test_tsscr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsscr()
