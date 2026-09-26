"""tssfe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssfe import tssfe


def test_tssfe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssfe()
