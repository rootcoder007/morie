"""spprlm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spprlm import spprlm


def test_spprlm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spprlm(y=None, X=None, W=None)
