"""wqas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqas import wqas


def test_wqas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqas()
