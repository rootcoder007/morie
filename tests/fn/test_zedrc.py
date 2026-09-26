"""zedrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zedrc import dose_resp_spatial


def test_zedrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dose_resp_spatial(data=None)
