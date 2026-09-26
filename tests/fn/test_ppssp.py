"""ppssp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppssp import ppssp


def test_ppssp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppssp()
