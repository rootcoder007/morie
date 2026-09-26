"""zxvnc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxvnc import vincenty_dist


def test_zxvnc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vincenty_dist(data=None)
