"""ptlfn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptlfn import l_function


def test_ptlfn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        l_function(data=None)
