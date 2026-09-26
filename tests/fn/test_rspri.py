"""rspri is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rspri import rspri


def test_rspri_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rspri()
