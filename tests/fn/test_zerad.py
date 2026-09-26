"""zerad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zerad import radiation_model


def test_zerad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        radiation_model(data=None)
