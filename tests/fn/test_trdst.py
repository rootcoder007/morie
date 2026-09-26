"""trdst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trdst import trdst


def test_trdst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trdst()
