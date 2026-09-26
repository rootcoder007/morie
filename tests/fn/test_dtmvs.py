"""dtmvs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtmvs import dtmvs


def test_dtmvs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtmvs()
