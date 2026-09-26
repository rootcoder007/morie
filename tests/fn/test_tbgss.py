"""tbgss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbgss import tbgss


def test_tbgss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbgss()
