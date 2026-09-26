"""dtscr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtscr import dtscr


def test_dtscr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtscr()
