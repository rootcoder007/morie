"""trtxi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trtxi import trtxi


def test_trtxi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trtxi()
