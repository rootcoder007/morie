"""tsmgw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsmgw import tsmgw


def test_tsmgw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsmgw()
