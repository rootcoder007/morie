"""svdcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdcm import svdcm


def test_svdcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svdcm()
