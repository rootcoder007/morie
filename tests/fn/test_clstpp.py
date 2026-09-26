"""clstpp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clstpp import clark_evans


def test_clstpp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clark_evans(coords=None)
