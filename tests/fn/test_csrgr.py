"""csrgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csrgr import csrgr


def test_csrgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csrgr()
