"""seexr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seexr import seexr


def test_seexr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seexr()
