"""csast is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csast import csast


def test_csast_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csast()
