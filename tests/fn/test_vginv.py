"""vginv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vginv import indicator_vario


def test_vginv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        indicator_vario(coords=None, values=None)
