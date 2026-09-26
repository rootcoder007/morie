"""wqefl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqefl import wqefl


def test_wqefl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqefl()
