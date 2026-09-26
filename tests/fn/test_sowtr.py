"""sowtr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sowtr import sowtr


def test_sowtr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sowtr()
