"""mtcrw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtcrw import mtcrw


def test_mtcrw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtcrw()
