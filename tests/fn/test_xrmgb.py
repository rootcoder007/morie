"""xrmgb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrmgb import mgwr_bandwidths


def test_xrmgb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mgwr_bandwidths(data=None)
