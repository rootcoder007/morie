"""stsmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stsmm import stsmm


def test_stsmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stsmm()
