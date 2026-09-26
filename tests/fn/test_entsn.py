"""entsn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.entsn import entsn


def test_entsn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        entsn()
