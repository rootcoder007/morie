"""svmp2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svmp2 import multiparty_2d


def test_svmp2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        multiparty_2d(data=None)
