"""tsslr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsslr import tsslr


def test_tsslr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsslr()
