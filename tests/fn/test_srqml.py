"""srqml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srqml import srqml


def test_srqml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srqml()
