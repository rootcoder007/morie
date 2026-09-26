"""dtrkn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtrkn import dtrkn


def test_dtrkn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtrkn()
