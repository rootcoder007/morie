"""xrwds is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwds import w_distance


def test_xrwds_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_distance(data=None)
