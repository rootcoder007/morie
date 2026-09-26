"""pttsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pttsp import pttsp


def test_pttsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pttsp()
