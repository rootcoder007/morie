"""zxfda is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxfda import fda_spatial


def test_zxfda_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fda_spatial(data=None)
