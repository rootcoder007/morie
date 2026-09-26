"""cosim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cosim import cosim


def test_cosim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cosim()
