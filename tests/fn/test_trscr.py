"""trscr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trscr import trscr


def test_trscr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trscr()
