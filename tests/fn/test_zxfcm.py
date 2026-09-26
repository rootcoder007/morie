"""zxfcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxfcm import fuzzy_cmeans_sp


def test_zxfcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fuzzy_cmeans_sp(data=None)
