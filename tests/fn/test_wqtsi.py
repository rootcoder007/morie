"""wqtsi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqtsi import wqtsi


def test_wqtsi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqtsi()
