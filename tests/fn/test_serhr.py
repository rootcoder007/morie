"""serhr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.serhr import serhr


def test_serhr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        serhr()
