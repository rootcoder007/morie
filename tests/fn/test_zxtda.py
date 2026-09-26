"""zxtda is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxtda import tda_persistent


def test_zxtda_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tda_persistent(data=None)
