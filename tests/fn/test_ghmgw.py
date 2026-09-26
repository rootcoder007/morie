"""ghmgw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghmgw import ghmgw


def test_ghmgw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghmgw()
