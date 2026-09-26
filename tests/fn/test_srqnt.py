"""srqnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srqnt import srqnt


def test_srqnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srqnt()
