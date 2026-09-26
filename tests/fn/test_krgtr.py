"""krgtr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.krgtr import krgtr


def test_krgtr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        krgtr()
