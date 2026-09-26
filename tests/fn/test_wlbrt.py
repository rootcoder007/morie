"""wlbrt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlbrt import wlbrt


def test_wlbrt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlbrt()
