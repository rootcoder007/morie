"""xrgr2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgr2 import gravity_poisson


def test_xrgr2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gravity_poisson(data=None)
