"""nnvor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nnvor import nnvor


def test_nnvor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nnvor()
