"""gaazbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gaazbr import gaazbr


def test_gaazbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gaazbr()
