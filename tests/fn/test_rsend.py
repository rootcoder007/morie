"""rsend is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsend import rsend


def test_rsend_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsend()
