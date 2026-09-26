"""aglst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aglst import aglst


def test_aglst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aglst()
