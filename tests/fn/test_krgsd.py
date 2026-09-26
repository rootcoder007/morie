"""krgsd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.krgsd import krgsd


def test_krgsd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        krgsd()
