"""nnextr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nnextr import nnextr


def test_nnextr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nnextr()
