"""rftrun is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rftrun import rftrun


def test_rftrun_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rftrun()
