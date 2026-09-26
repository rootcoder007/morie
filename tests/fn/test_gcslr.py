"""gcslr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcslr import gcslr


def test_gcslr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcslr()
