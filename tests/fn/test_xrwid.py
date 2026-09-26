"""xrwid is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwid import w_inverse_dist


def test_xrwid_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_inverse_dist(data=None)
