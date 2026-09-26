"""gerdx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gerdx import gerdx


def test_gerdx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gerdx()
