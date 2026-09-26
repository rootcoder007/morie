"""gcdrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcdrg import gcdrg


def test_gcdrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcdrg()
