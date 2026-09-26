"""rgfsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rgfsp import rgfsp


def test_rgfsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rgfsp()
