"""gdsex is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdsex import gdsex


def test_gdsex_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdsex()
