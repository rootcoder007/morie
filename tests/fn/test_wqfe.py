"""wqfe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqfe import wqfe


def test_wqfe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqfe()
