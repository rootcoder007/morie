"""wqph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqph import wqph


def test_wqph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqph()
