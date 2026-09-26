"""hytwi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hytwi import hytwi


def test_hytwi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hytwi()
