"""xrmgw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrmgw import mgwr_estimate


def test_xrmgw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mgwr_estimate(data=None)
