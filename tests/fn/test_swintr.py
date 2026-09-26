"""swintr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swintr import swin_transformer


def test_swintr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swin_transformer(x=None, window_size=None)
