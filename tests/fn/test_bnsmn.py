"""bnsmn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bnsmn import bnsmn


def test_bnsmn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bnsmn()
