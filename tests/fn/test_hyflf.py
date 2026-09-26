"""hyflf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyflf import hyflf


def test_hyflf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyflf()
