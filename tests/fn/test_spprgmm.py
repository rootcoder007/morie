"""spprgmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spprgmm import spprgmm


def test_spprgmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spprgmm(y=None, X=None, W=None)
