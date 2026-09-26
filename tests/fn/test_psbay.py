"""psbay is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psbay import psbay


def test_psbay_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        psbay()
