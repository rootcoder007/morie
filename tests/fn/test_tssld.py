"""tssld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssld import tssld


def test_tssld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssld()
