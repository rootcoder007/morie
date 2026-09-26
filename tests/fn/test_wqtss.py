"""wqtss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqtss import wqtss


def test_wqtss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqtss()
