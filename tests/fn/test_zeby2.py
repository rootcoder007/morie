"""zeby2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zeby2 import bym2_model


def test_zeby2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bym2_model(data=None)
