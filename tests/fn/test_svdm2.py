"""svdm2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdm2 import dim_test_2


def test_svdm2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dim_test_2(data=None)
