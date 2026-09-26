"""bkflt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bkflt import bkflt


def test_bkflt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bkflt()
