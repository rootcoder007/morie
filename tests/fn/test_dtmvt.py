"""dtmvt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtmvt import dtmvt


def test_dtmvt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtmvt()
